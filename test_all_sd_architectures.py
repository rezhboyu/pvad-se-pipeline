#!/usr/bin/env python3
"""
Speaker-Dependent 多架構比較實驗（含噪音增強）
================================================
架構:
  1. MLP (256-128-64)
  2. LSTM (Bi-LSTM 128x2)
  3. GRU + Attention (Bi-GRU 128 + Self-Attention)
  4. CNN (1D Conv 64-128-256)
  5. CNN + Self-Attention
  6. SVM (RBF)
  7. Random Forest

所有模型使用噪音增強訓練 (SNR 5/10/15/20 dB)
特徵: 45-dim SDMFCC (15 MFCC + 15 Delta + 15 Delta2)
正樣本: hsuan 語音幀 | 負樣本: 0911+FEMH 語音幀 + 靜音
"""

import sys, json, time, numpy as np, librosa
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.audio import read_audio, SAMPLE_RATE

MAT_DIR = Path.home() / "Desktop" / "VOICE" / "MAT"
PROJECT_DIR = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════
# 特徵提取
# ══════════════════════════════════════════════════════

def extract_sdmfcc(audio, sr=16000, hop_length=512):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T

def energy_vad(audio, hop=512, thr=0.01):
    n = (len(audio) - hop) // hop + 1
    return np.array([1 if np.sqrt(np.mean(audio[i*hop:i*hop+hop]**2)) > thr else 0
                     for i in range(n)], dtype=np.int64)

def add_noise(audio, snr_db):
    noise = np.random.randn(len(audio)).astype(np.float32)
    sp = np.mean(audio**2)
    np_ = sp / (10 ** (snr_db / 10))
    return audio + noise * np.sqrt(np_ / (np.mean(noise**2) + 1e-12))


# ══════════════════════════════════════════════════════
# 模型定義
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
    """Bi-GRU + Self-Attention (inspired by MDPI 2025 PVAD paper)"""
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


class SD_CNN(nn.Module):
    def __init__(self, dim=45):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, 2)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 45)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class SD_CNN_SelfAttention(nn.Module):
    """CNN + Self-Attention (inspired by arxiv 2203.02944)"""
    def __init__(self, dim=45):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 45)
        x = self.conv(x)     # (B, 128, 45)
        x = x.permute(0, 2, 1)  # (B, 45, 128)
        a, _ = self.attn(x, x, x)
        x = self.norm(x + a)
        x = x.permute(0, 2, 1)  # (B, 128, 45)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ══════════════════════════════════════════════════════
# 訓練
# ══════════════════════════════════════════════════════

def prepare_data():
    hsuan_files = sorted(MAT_DIR.glob("*hsuan*.wav"))
    interf_files = sorted(MAT_DIR.glob("*0911*.wav"))
    femh_files = sorted(MAT_DIR.glob("*FEMH*.wav"))
    snrs = [5, 10, 15, 20]

    all_X, all_y = [], []
    for f in hsuan_files:
        a = read_audio(str(f))
        feat = extract_sdmfcc(a); vad = energy_vad(a)
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

    X = np.concatenate(all_X); y = np.concatenate(all_y)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-8
    return (X_tr - mean) / std, (X_val - mean) / std, y_tr, y_val, mean, std


def train_torch(model, X_tr, X_val, y_tr, y_val, epochs=60, lr=1e-3, name="model"):
    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), batch_size=256, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=512)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best_acc, best_st = 0, None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                c += (model(xb).argmax(1) == yb).sum().item(); t += len(yb)
        acc = c / t; sch.step(1 - acc)
        if acc > best_acc: best_acc = acc; best_st = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"    [{name}] ep{ep+1}: val_acc={acc:.3f} (best={best_acc:.3f})")

    model.load_state_dict(best_st)
    params = sum(p.numel() for p in model.parameters())
    print(f"    [{name}] DONE: best={best_acc:.3f}, params={params:,}")
    return model, best_acc, params


# ══════════════════════════════════════════════════════
# 測試
# ══════════════════════════════════════════════════════

def build_scenarios():
    hs = [read_audio(str(f)) for f in sorted(MAT_DIR.glob("*hsuan_7.wav"))[:3]]
    ia = [read_audio(str(f)) for f in sorted(MAT_DIR.glob("*0911*_7.wav"))[:2]]
    fa = read_audio(str(sorted(MAT_DIR.glob("*FEMH_7.wav"))[0]))
    sl = int(2.5 * SAMPLE_RATE)
    g = lambda a, i: a[i % len(a)][:sl]

    sa = np.concatenate([g(hs,0), g(ia,0), g(hs,1), g(ia,1), g(hs,2)])
    sd = np.concatenate([g(hs,0), fa[:sl], g(hs,1), fa[sl:2*sl], g(hs,2)])
    n = np.random.RandomState(42).randn(len(sa)).astype(np.float32)
    n *= np.sqrt(np.mean(sa**2) / 10 / (np.mean(n**2) + 1e-12))
    sc = sa + n
    ol = int(3.0 * SAMPLE_RATE)
    sb = np.concatenate([g(hs,0), hs[0][sl:sl+ol] + ia[0][:ol], g(hs,1)])

    labs = [("hsuan_0.0-2.5s",0,2.5,True),("interferer_2.5-5.0s",2.5,5,False),
            ("hsuan_5.0-7.5s",5,7.5,True),("interferer_7.5-10.0s",7.5,10,False),
            ("hsuan_10.0-12.5s",10,12.5,True)]
    labs_b = [("hsuan_only_0.0-2.5s",0,2.5,True),("hsuan+interferer_2.5-5.5s",2.5,5.5,True),
              ("hsuan_only_5.5-8.0s",5.5,8,True)]
    return {"scenario_a":(sa,labs),"scenario_b":(sb,labs_b),"scenario_c":(sc,labs),"scenario_d":(sd,labs)}


def evaluate_all(models_dict, scenarios, mean, std, pvad_metrics):
    hop = 512; fd = hop / SAMPLE_RATE
    all_results = {}

    for mn, minfo in models_dict.items():
        all_results[mn] = {"per_scenario": {}, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for sc_name, (audio, labels) in scenarios.items():
            feat = (extract_sdmfcc(audio) - mean) / std
            if minfo["type"] == "torch":
                minfo["model"].eval()
                with torch.no_grad():
                    preds = minfo["model"](torch.FloatTensor(feat)).argmax(1).numpy()
            else:
                preds = minfo["model"].predict(feat)

            sc_res = {}
            for sn, ss, se, is_t in labels:
                sf = int(ss/fd); ef = min(int(se/fd), len(preds))
                tr = float(np.mean(preds[sf:ef]))
                sc_res[sn] = {"target_ratio": round(tr, 4), "is_target": is_t, "n_frames": ef-sf}
                if sc_name != "scenario_b":
                    t = int(tr * (ef-sf)); nt = (ef-sf) - t
                    if is_t: all_results[mn]["tp"] += t; all_results[mn]["fn"] += nt
                    else: all_results[mn]["fp"] += t; all_results[mn]["tn"] += nt
            all_results[mn]["per_scenario"][sc_name] = sc_res

        tp, fn, fp, tn = all_results[mn]["tp"], all_results[mn]["fn"], all_results[mn]["fp"], all_results[mn]["tn"]
        p = tp/(tp+fp) if (tp+fp) else 0
        r = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
        all_results[mn].update({"accuracy": acc, "precision": p, "recall": r, "f1": f1,
                                "val_acc": minfo.get("val_acc", 0), "params": minfo.get("params", "N/A")})
    return all_results


def main():
    print("=" * 80)
    print("Speaker-Dependent 多架構比較 (噪音增強訓練)")
    print("=" * 80)

    X_tr, X_val, y_tr, y_val, mean, std = prepare_data()
    print(f"訓練: {len(X_tr)} frames, 驗證: {len(X_val)} frames")

    models = {}
    print("\n=== 訓練 PyTorch 模型 ===")

    for name, cls in [
        ("MLP", SD_MLP),
        ("BiLSTM", SD_BiLSTM),
        ("BiGRU_Attn", SD_BiGRU_Attention),
        ("CNN", SD_CNN),
    ]:
        t0 = time.time()
        m = cls()
        m, vacc, params = train_torch(m, X_tr, X_val, y_tr, y_val, epochs=60, name=name)
        elapsed = time.time() - t0
        models[name] = {"model": m, "type": "torch", "val_acc": vacc, "params": params, "time": elapsed}

    print("\n=== 訓練 sklearn 模型 ===")

    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    t0 = time.time()
    svm_base = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
    svm_base.fit(X_tr, y_tr)
    svm_acc = accuracy_score(y_val, svm_base.predict(X_val))
    models["LinearSVM"] = {"model": svm_base, "type": "sklearn", "val_acc": svm_acc, "params": "N/A", "time": time.time()-t0}
    print(f"    [LinearSVM] val_acc={svm_acc:.3f}, time={models['LinearSVM']['time']:.1f}s")

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42)
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_val, rf.predict(X_val))
    models["RF"] = {"model": rf, "type": "sklearn", "val_acc": rf_acc, "params": "N/A", "time": time.time()-t0}
    print(f"    [RF] val_acc={rf_acc:.3f}, time={models['RF']['time']:.1f}s")

    # 測試
    print("\n=== 四場景測試 ===")
    scenarios = build_scenarios()

    with open(PROJECT_DIR / "test_parallel/test_report_parallel.json", encoding="utf-8") as f:
        pvad_m = json.load(f)["metrics"]

    results = evaluate_all(models, scenarios, mean, std, pvad_m)

    # ══════ 打印結果 ══════
    mnames = list(models.keys())

    # Per-scenario table
    print("\n" + "=" * 120)
    print("TARGET RATIO 比較")
    print("=" * 120)
    h = f"{'Segment':>28s}"
    for mn in mnames: h += f" {mn:>10s}"
    h += f" {'pVAD-SE':>10s} {'GT':>4s}"
    print(h); print("-" * len(h))

    for sc_name in ["scenario_a", "scenario_c", "scenario_d"]:
        print(f"\n  [{sc_name.upper()}]")
        first = results[mnames[0]]["per_scenario"][sc_name]
        for sn in first:
            gt = "T" if first[sn]["is_target"] else "I"
            line = f"  {sn:>28s}"
            for mn in mnames:
                tr = results[mn]["per_scenario"][sc_name][sn]["target_ratio"]
                line += f" {tr:9.1%}"
            pvad_tr = pvad_m.get(sc_name, {}).get(sn, {}).get("target_ratio", -1)
            line += f" {pvad_tr:9.1%}" if pvad_tr >= 0 else f" {'N/A':>10s}"
            line += f" {gt:>4s}"
            print(line)

    # F1 summary
    print("\n" + "=" * 100)
    print("F1 SCORE 排名 (場景 A+C+D, 噪音增強訓練)")
    print("=" * 100)
    print(f"{'Rank':>4s} {'Model':>12s} {'Val Acc':>8s} {'Params':>10s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Train':>7s}")
    print("-" * 80)

    sorted_models = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for rank, (mn, r) in enumerate(sorted_models, 1):
        params_s = f"{r['params']:,}" if isinstance(r["params"], int) else r["params"]
        train_t = f"{models[mn]['time']:.0f}s"
        print(f"{rank:>4d} {mn:>12s} {r['val_acc']:7.1%} {params_s:>10s} {r['accuracy']:7.1%} {r['precision']:7.1%} {r['recall']:7.1%} {r['f1']:7.3f} {train_t:>7s}")

    print(f"   - {'pVAD-SE':>12s} {'N/A':>8s} {'6.9M+83K':>10s} {'65.0%':>8s} {'76.8%':>8s} {'59.8%':>8s} {'0.672':>8s} {'N/A':>7s}")

    # 存檔
    save_data = {"experiment": "SD multi-arch comparison (noise-augmented)", "models": {}}
    for mn, r in results.items():
        save_data["models"][mn] = {
            "val_accuracy": round(r["val_acc"], 4),
            "params": r["params"],
            "test_metrics": {
                "accuracy": round(r["accuracy"], 4), "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4), "f1": round(r["f1"], 4),
            },
            "confusion_matrix": {"tp": r["tp"], "fn": r["fn"], "fp": r["fp"], "tn": r["tn"]},
            "per_scenario": r["per_scenario"],
        }

    out = PROJECT_DIR / "test_parallel" / "sd_all_architectures_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n報告已存: {out}")


if __name__ == "__main__":
    main()
