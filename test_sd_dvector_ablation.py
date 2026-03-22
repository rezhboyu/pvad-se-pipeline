#!/usr/bin/env python3
"""
Speaker-Dependent 特徵消融實驗：SDMFCC vs d-vector + WeSpeaker vs CAM++
========================================================================
消融維度:
  1. 特徵: 45-dim SDMFCC vs 256-dim WeSpeaker d-vector vs 512-dim CAM++ d-vector
  2. 架構: MLP / BiLSTM / BiGRU_Attn / LinearSVM / RandomForest (不含 CNN)
  3. 噪音增強: SNR 5/10/15/20 dB

d-vector 幀級特徵提取方式:
  - 用滑動窗口 (0.5s, hop=32ms) 從音頻中逐段提取 speaker embedding
  - 每個窗口產生一個 d-vector 作為該時段的特徵
  - 標註: energy VAD 標記目標 vs 非目標

用法:
    python test_sd_dvector_ablation.py
"""

import sys
import json
import time
import numpy as np
import librosa
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.audio import read_audio, SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder, _compute_fbank

PROJECT_DIR = Path(__file__).resolve().parent
MAT_DIR = Path.home() / "Desktop" / "VOICE" / "MAT"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "test_sd_dvector_ablation"


# ══════════════════════════════════════════════════════
# 特徵提取
# ══════════════════════════════════════════════════════

def extract_sdmfcc(audio, sr=16000, hop_length=512):
    """45-dim SDMFCC: 15 MFCC + 15 Delta + 15 Delta2"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T  # (T, 45)


def extract_dvector_frames(audio, encoder, window_sec=0.5, hop_sec=0.032):
    """
    用滑動窗口從音頻提取幀級 d-vector 特徵。

    Parameters
    ----------
    audio : np.ndarray, 16kHz mono
    encoder : SpeakerEncoder (WeSpeaker or CAM++)
    window_sec : float, 提取窗口長度（秒）
    hop_sec : float, 幀移（秒），對齊 SDMFCC 的 hop_length=512 (32ms)

    Returns
    -------
    features : (N, embed_dim) float32
    """
    win_samples = int(window_sec * SAMPLE_RATE)
    hop_samples = int(hop_sec * SAMPLE_RATE)  # 512 samples = 32ms
    n_frames = max(1, (len(audio) - win_samples) // hop_samples + 1)

    features = []
    for i in range(n_frames):
        start = i * hop_samples
        end = start + win_samples
        if end > len(audio):
            # 最後不足一窗的部分，用零補齊
            chunk = np.zeros(win_samples, dtype=np.float32)
            chunk[:len(audio) - start] = audio[start:]
        else:
            chunk = audio[start:end]

        emb = encoder.extract_embedding(chunk)
        features.append(emb)

    return np.array(features, dtype=np.float32)  # (N, embed_dim)


def energy_vad(audio, hop=512, thr=0.01):
    """簡易能量 VAD 標註"""
    n = (len(audio) - hop) // hop + 1
    return np.array([1 if np.sqrt(np.mean(audio[i*hop:i*hop+hop]**2)) > thr else 0
                     for i in range(n)], dtype=np.int64)


def add_noise(audio, snr_db):
    """加入高斯噪聲"""
    noise = np.random.randn(len(audio)).astype(np.float32)
    sp = np.mean(audio**2)
    np_ = sp / (10 ** (snr_db / 10))
    return audio + noise * np.sqrt(np_ / (np.mean(noise**2) + 1e-12))


# ══════════════════════════════════════════════════════
# 模型定義（去掉 CNN，只保留 MLP/BiLSTM/BiGRU_Attn）
# ══════════════════════════════════════════════════════

class SD_MLP(nn.Module):
    def __init__(self, dim=45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)


class SD_BiLSTM(nn.Module):
    def __init__(self, dim=45, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden * 2, 2)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        o, _ = self.lstm(x)
        return self.fc(self.drop(o.squeeze(1)))


class SD_BiGRU_Attention(nn.Module):
    def __init__(self, dim=45, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.MultiheadAttention(hidden * 2, num_heads=4, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden * 2, 2)
        self.drop = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden * 2)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        o, _ = self.gru(x)
        attn_out, _ = self.attn(o, o, o)
        o = self.norm(o + attn_out)
        return self.fc(self.drop(o.squeeze(1)))


# ══════════════════════════════════════════════════════
# 數據準備
# ══════════════════════════════════════════════════════

def prepare_data_sdmfcc():
    """原版 SDMFCC 特徵 (45-dim)"""
    hsuan_files = sorted(MAT_DIR.glob("*hsuan*.wav"))
    interf_files = sorted(MAT_DIR.glob("*0911*.wav"))
    femh_files = sorted(MAT_DIR.glob("*FEMH*.wav"))
    snrs = [5, 10, 15, 20]

    all_X, all_y = [], []
    for f in hsuan_files:
        a = read_audio(str(f))
        feat = extract_sdmfcc(a)
        vad = energy_vad(a)
        ml = min(len(feat), len(vad))
        all_X.append(feat[:ml]); all_y.append(vad[:ml])
        for snr in snrs:
            fn = extract_sdmfcc(add_noise(a, snr))
            ml_n = min(len(fn), ml)
            all_X.append(fn[:ml_n]); all_y.append(vad[:ml_n])

    for f in list(interf_files) + list(femh_files):
        a = read_audio(str(f))
        feat = extract_sdmfcc(a)
        all_X.append(feat); all_y.append(np.zeros(len(feat), dtype=np.int64))
        for snr in snrs:
            fn = extract_sdmfcc(add_noise(a, snr))
            all_X.append(fn); all_y.append(np.zeros(len(fn), dtype=np.int64))

    return _split_and_normalize(all_X, all_y)


def prepare_data_dvector(encoder, encoder_name="dvector"):
    """d-vector 特徵提取（WeSpeaker 256-dim 或 CAM++ 512-dim）"""
    hsuan_files = sorted(MAT_DIR.glob("*hsuan*.wav"))
    interf_files = sorted(MAT_DIR.glob("*0911*.wav"))
    femh_files = sorted(MAT_DIR.glob("*FEMH*.wav"))
    snrs = [5, 10, 15, 20]

    print(f"    [{encoder_name}] 提取 d-vector 幀級特徵（這可能需要較長時間）...")
    all_X, all_y = [], []
    total_files = len(hsuan_files) + len(interf_files) + len(femh_files)
    file_idx = 0

    for f in hsuan_files:
        file_idx += 1
        a = read_audio(str(f))
        vad = energy_vad(a)
        # 原始音頻
        feat = extract_dvector_frames(a, encoder)
        ml = min(len(feat), len(vad))
        all_X.append(feat[:ml]); all_y.append(vad[:ml])
        # 噪音增強
        for snr in snrs:
            noisy = add_noise(a, snr)
            fn = extract_dvector_frames(noisy, encoder)
            ml_n = min(len(fn), ml)
            all_X.append(fn[:ml_n]); all_y.append(vad[:ml_n])
        print(f"      [{file_idx}/{total_files}] {f.name} done (+ {len(snrs)} augments)")

    for f in list(interf_files) + list(femh_files):
        file_idx += 1
        a = read_audio(str(f))
        # 原始音頻
        feat = extract_dvector_frames(a, encoder)
        all_X.append(feat); all_y.append(np.zeros(len(feat), dtype=np.int64))
        # 噪音增強
        for snr in snrs:
            noisy = add_noise(a, snr)
            fn = extract_dvector_frames(noisy, encoder)
            all_X.append(fn); all_y.append(np.zeros(len(fn), dtype=np.int64))
        print(f"      [{file_idx}/{total_files}] {f.name} done (+ {len(snrs)} augments)")

    return _split_and_normalize(all_X, all_y)


def _split_and_normalize(all_X, all_y):
    """共用的數據分割與正規化"""
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mean = X_tr.mean(0)
    std = X_tr.std(0) + 1e-8
    return (X_tr - mean) / std, (X_val - mean) / std, y_tr, y_val, mean, std


# ══════════════════════════════════════════════════════
# 訓練
# ══════════════════════════════════════════════════════

def train_torch(model, X_tr, X_val, y_tr, y_val, epochs=60, lr=1e-3, name="model"):
    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=256, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=512
    )
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best_acc, best_st = 0, None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                c += (model(xb).argmax(1) == yb).sum().item()
                t += len(yb)
        acc = c / t
        sch.step(1 - acc)
        if acc > best_acc:
            best_acc = acc
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"      [{name}] ep{ep+1}: val_acc={acc:.3f} (best={best_acc:.3f})")

    model.load_state_dict(best_st)
    params = sum(p.numel() for p in model.parameters())
    print(f"      [{name}] DONE: best={best_acc:.3f}, params={params:,}")
    return model, best_acc, params


# ══════════════════════════════════════════════════════
# 測試場景
# ══════════════════════════════════════════════════════

def build_scenarios():
    """建立 4 個測試場景（與原版相同）"""
    hs = [read_audio(str(f)) for f in sorted(MAT_DIR.glob("*hsuan_7.wav"))[:3]]
    ia = [read_audio(str(f)) for f in sorted(MAT_DIR.glob("*0911*_7.wav"))[:2]]
    fa = read_audio(str(sorted(MAT_DIR.glob("*FEMH_7.wav"))[0]))
    sl = int(2.5 * SAMPLE_RATE)
    g = lambda a, i: a[i % len(a)][:sl]

    sa = np.concatenate([g(hs, 0), g(ia, 0), g(hs, 1), g(ia, 1), g(hs, 2)])
    sd = np.concatenate([g(hs, 0), fa[:sl], g(hs, 1), fa[sl:2*sl], g(hs, 2)])
    n = np.random.RandomState(42).randn(len(sa)).astype(np.float32)
    n *= np.sqrt(np.mean(sa**2) / 10 / (np.mean(n**2) + 1e-12))
    sc = sa + n
    ol = int(3.0 * SAMPLE_RATE)
    sb = np.concatenate([g(hs, 0), hs[0][sl:sl+ol] + ia[0][:ol], g(hs, 1)])

    labs = [
        ("hsuan_0.0-2.5s", 0, 2.5, True),
        ("interferer_2.5-5.0s", 2.5, 5, False),
        ("hsuan_5.0-7.5s", 5, 7.5, True),
        ("interferer_7.5-10.0s", 7.5, 10, False),
        ("hsuan_10.0-12.5s", 10, 12.5, True),
    ]
    labs_b = [
        ("hsuan_only_0.0-2.5s", 0, 2.5, True),
        ("hsuan+interferer_2.5-5.5s", 2.5, 5.5, True),
        ("hsuan_only_5.5-8.0s", 5.5, 8, True),
    ]
    return {
        "scenario_a": (sa, labs),
        "scenario_b": (sb, labs_b),
        "scenario_c": (sc, labs),
        "scenario_d": (sd, labs),
    }


def evaluate_sdmfcc(models_dict, scenarios, mean, std):
    """用 SDMFCC 特徵評估"""
    hop = 512
    fd = hop / SAMPLE_RATE
    return _evaluate(models_dict, scenarios, mean, std,
                     feat_fn=lambda audio, m, s: (extract_sdmfcc(audio) - m) / s,
                     fd=fd)


def evaluate_dvector(models_dict, scenarios, mean, std, encoder):
    """用 d-vector 特徵評估"""
    hop = 512
    fd = hop / SAMPLE_RATE
    return _evaluate(models_dict, scenarios, mean, std,
                     feat_fn=lambda audio, m, s: (extract_dvector_frames(audio, encoder) - m) / s,
                     fd=fd)


def _evaluate(models_dict, scenarios, mean, std, feat_fn, fd):
    """通用評估函數"""
    all_results = {}

    for mn, minfo in models_dict.items():
        all_results[mn] = {"per_scenario": {}, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for sc_name, (audio, labels) in scenarios.items():
            feat = feat_fn(audio, mean, std)

            if minfo["type"] == "torch":
                minfo["model"].eval()
                with torch.no_grad():
                    preds = minfo["model"](torch.FloatTensor(feat)).argmax(1).numpy()
            else:
                preds = minfo["model"].predict(feat)

            sc_res = {}
            for sn, ss, se, is_t in labels:
                sf = int(ss / fd)
                ef = min(int(se / fd), len(preds))
                tr = float(np.mean(preds[sf:ef]))
                sc_res[sn] = {
                    "target_ratio": round(tr, 4),
                    "is_target": is_t,
                    "n_frames": ef - sf,
                }
                if sc_name != "scenario_b":
                    t = int(tr * (ef - sf))
                    nt = (ef - sf) - t
                    if is_t:
                        all_results[mn]["tp"] += t
                        all_results[mn]["fn"] += nt
                    else:
                        all_results[mn]["fp"] += t
                        all_results[mn]["tn"] += nt
            all_results[mn]["per_scenario"][sc_name] = sc_res

        tp = all_results[mn]["tp"]
        fn = all_results[mn]["fn"]
        fp = all_results[mn]["fp"]
        tn = all_results[mn]["tn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
        all_results[mn].update({
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "val_acc": minfo.get("val_acc", 0),
            "params": minfo.get("params", "N/A"),
        })
    return all_results


# ══════════════════════════════════════════════════════
# 訓練 + 測試一組特徵
# ══════════════════════════════════════════════════════

def train_and_evaluate(feat_name, feat_dim, X_tr, X_val, y_tr, y_val, mean, std,
                       scenarios, eval_fn):
    """訓練所有架構並評估，回傳結果字典"""
    print(f"\n  === 訓練 {feat_name} 模型 (dim={feat_dim}) ===")
    models = {}

    # PyTorch 模型
    for name, cls in [
        ("MLP", SD_MLP),
        ("BiLSTM", SD_BiLSTM),
        ("BiGRU_Attn", SD_BiGRU_Attention),
    ]:
        t0 = time.time()
        m = cls(dim=feat_dim)
        m, vacc, params = train_torch(m, X_tr, X_val, y_tr, y_val, epochs=60, name=name)
        elapsed = time.time() - t0
        models[name] = {
            "model": m, "type": "torch",
            "val_acc": vacc, "params": params, "time": elapsed,
        }

    # sklearn 模型
    print(f"    訓練 sklearn 模型...")
    t0 = time.time()
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
    svm.fit(X_tr, y_tr)
    svm_acc = accuracy_score(y_val, svm.predict(X_val))
    models["LinearSVM"] = {
        "model": svm, "type": "sklearn",
        "val_acc": svm_acc, "params": "N/A", "time": time.time() - t0,
    }
    print(f"      [LinearSVM] val_acc={svm_acc:.3f}")

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15,
        class_weight="balanced", random_state=42
    )
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_val, rf.predict(X_val))
    models["RF"] = {
        "model": rf, "type": "sklearn",
        "val_acc": rf_acc, "params": "N/A", "time": time.time() - t0,
    }
    print(f"      [RF] val_acc={rf_acc:.3f}")

    # 評估
    print(f"\n  === 評估 {feat_name} ===")
    results = eval_fn(models, scenarios, mean, std)
    return results, models


# ══════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Speaker-Dependent 特徵消融: SDMFCC vs d-vector (WeSpeaker / CAM++)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 載入 speaker encoder ──
    print("\n[1] 載入 Speaker Encoder 模型...")
    encoders = {}

    wespeaker_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if wespeaker_path.exists():
        encoders["WeSpeaker_256d"] = SpeakerEncoder(str(wespeaker_path))
        print(f"  WeSpeaker: embed_dim={encoders['WeSpeaker_256d'].embed_dim}")

    campp_path = MODELS_DIR / "campplus" / "campplus.onnx"
    if campp_path.exists():
        encoders["CAM++_512d"] = SpeakerEncoder(str(campp_path))
        print(f"  CAM++: embed_dim={encoders['CAM++_512d'].embed_dim}")

    if not encoders:
        print("  [ERROR] 沒有找到任何 speaker encoder 模型！")
        print(f"  請確認以下路徑存在:")
        print(f"    - {wespeaker_path}")
        print(f"    - {campp_path}")
        return

    # ── 建立測試場景 ──
    print("\n[2] 建立測試場景...")
    scenarios = build_scenarios()
    print(f"  場景數: {len(scenarios)}")

    all_ablation_results = {}

    # ── 消融 1: SDMFCC baseline ──
    print("\n" + "=" * 60)
    print("[3] 消融 1: SDMFCC (45-dim) — baseline")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, mean, std = prepare_data_sdmfcc()
    print(f"  訓練: {len(X_tr)} frames, 驗證: {len(X_val)} frames, dim={X_tr.shape[1]}")

    results_sdmfcc, _ = train_and_evaluate(
        "SDMFCC_45d", 45, X_tr, X_val, y_tr, y_val, mean, std,
        scenarios,
        eval_fn=evaluate_sdmfcc,
    )
    all_ablation_results["SDMFCC_45d"] = results_sdmfcc

    # ── 消融 2+: d-vector 特徵（每個 encoder 一組）──
    for enc_name, encoder in encoders.items():
        embed_dim = encoder.embed_dim
        print(f"\n{'=' * 60}")
        print(f"[{3 + list(encoders.keys()).index(enc_name) + 1}] "
              f"消融: {enc_name} ({embed_dim}-dim d-vector)")
        print(f"{'=' * 60}")

        X_tr, X_val, y_tr, y_val, mean, std = prepare_data_dvector(encoder, enc_name)
        print(f"  訓練: {len(X_tr)} frames, 驗證: {len(X_val)} frames, dim={X_tr.shape[1]}")

        results_dv, _ = train_and_evaluate(
            enc_name, embed_dim, X_tr, X_val, y_tr, y_val, mean, std,
            scenarios,
            eval_fn=lambda models, sc, m, s, enc=encoder: evaluate_dvector(models, sc, m, s, enc),
        )
        all_ablation_results[enc_name] = results_dv

    # ══════════════════════════════════════════════════
    # 打印結果
    # ══════════════════════════════════════════════════
    print("\n\n" + "=" * 120)
    print("消融結果匯總")
    print("=" * 120)

    # F1 排名表
    print(f"\n{'Feature':>16s} {'Model':>12s} {'Val Acc':>8s} {'Params':>10s} "
          f"{'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print("-" * 95)

    ranked = []
    for feat_name, feat_results in all_ablation_results.items():
        for model_name, r in feat_results.items():
            ranked.append((feat_name, model_name, r))

    ranked.sort(key=lambda x: x[2]["f1"], reverse=True)

    for feat_name, model_name, r in ranked:
        params_s = f"{r['params']:,}" if isinstance(r["params"], int) else r["params"]
        print(f"{feat_name:>16s} {model_name:>12s} {r['val_acc']:7.1%} {params_s:>10s} "
              f"{r['accuracy']:7.1%} {r['precision']:7.1%} {r['recall']:7.1%} {r['f1']:7.3f}")

    # pVAD-SE 參考
    print(f"{'pVAD-SE':>16s} {'(ref)':>12s} {'N/A':>8s} {'6.9M+83K':>10s} "
          f"{'65.0%':>8s} {'76.8%':>8s} {'59.8%':>8s} {'0.672':>8s}")

    # Per-scenario 比較（只顯示每種特徵的最佳模型）
    print(f"\n\n{'=' * 120}")
    print("每種特徵的最佳模型 — 各場景 Target Ratio")
    print("=" * 120)

    for feat_name, feat_results in all_ablation_results.items():
        best_model = max(feat_results.items(), key=lambda x: x[1]["f1"])
        mn, r = best_model
        print(f"\n  {feat_name} → 最佳: {mn} (F1={r['f1']:.3f})")
        for sc_name in ["scenario_a", "scenario_c", "scenario_d"]:
            sc_data = r["per_scenario"].get(sc_name, {})
            line = f"    [{sc_name}] "
            for sn, info in sc_data.items():
                gt = "T" if info["is_target"] else "I"
                line += f"{sn}: {info['target_ratio']:.1%}({gt})  "
            print(line)

    # 特徵間差異分析
    print(f"\n\n{'=' * 120}")
    print("特徵消融分析")
    print("=" * 120)

    feat_best_f1 = {}
    for feat_name, feat_results in all_ablation_results.items():
        best = max(feat_results.values(), key=lambda x: x["f1"])
        feat_best_f1[feat_name] = best["f1"]

    baseline_f1 = feat_best_f1.get("SDMFCC_45d", 0)
    for feat_name, f1 in sorted(feat_best_f1.items(), key=lambda x: x[1], reverse=True):
        delta = f1 - baseline_f1
        arrow = "+" if delta > 0 else ""
        print(f"  {feat_name:>16s}: best F1 = {f1:.3f}  ({arrow}{delta:.3f} vs SDMFCC)")

    # ── 存檔 ──
    save_data = {
        "experiment": "SD feature ablation: SDMFCC vs d-vector (WeSpeaker / CAM++)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architectures": ["MLP", "BiLSTM", "BiGRU_Attn", "LinearSVM", "RF"],
        "features": list(all_ablation_results.keys()),
        "results": {},
    }

    for feat_name, feat_results in all_ablation_results.items():
        save_data["results"][feat_name] = {}
        for mn, r in feat_results.items():
            save_data["results"][feat_name][mn] = {
                "val_accuracy": round(r["val_acc"], 4),
                "params": r["params"],
                "test_metrics": {
                    "accuracy": round(r["accuracy"], 4),
                    "precision": round(r["precision"], 4),
                    "recall": round(r["recall"], 4),
                    "f1": round(r["f1"], 4),
                },
                "confusion_matrix": {
                    "tp": r["tp"], "fn": r["fn"],
                    "fp": r["fp"], "tn": r["tn"],
                },
                "per_scenario": r["per_scenario"],
            }

    out_path = OUTPUT_DIR / "sd_dvector_ablation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n報告已存: {out_path}")


if __name__ == "__main__":
    main()
