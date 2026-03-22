#!/usr/bin/env python3
"""
Speaker-Dependent SDMFCC MLP 訓練 & 測試
==========================================
只用 hsuan 的音檔訓練，測試同樣的四場景。

訓練策略:
  - Positive (1): hsuan 語音幀（高能量段）
  - Negative (0): hsuan 靜音幀（低能量段）
  - 特徵: 45-dim SDMFCC (15 MFCC + 15 Delta + 15 Delta2)
  - 模型: MLP 256-128-64, dropout=0.3

因為只用 hsuan 訓練，模型學到的是「hsuan 的語音特徵 vs 靜音」。
理論上，其他人的語音如果 MFCC 特徵與 hsuan 不同，也應該被判為 0。
但 MFCC 主要捕捉語音內容特徵而非說話者特徵，所以可能會誤判。
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

VOICE_DIR = Path.home() / "Desktop" / "VOICE"
MAT_DIR = VOICE_DIR / "MAT"

from utils.audio import read_audio, SAMPLE_RATE


# ══════════════════════════════════════════════════════
# 1. SDMFCC 特徵提取
# ══════════════════════════════════════════════════════

def extract_sdmfcc(audio, sr=16000, hop_length=512):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T  # (T, 45)


def energy_vad_labels(audio, hop_length=512, threshold=0.01):
    """用能量做簡易 VAD 標註：能量 > threshold → speech(1), 否則 silence(0)"""
    n_frames = (len(audio) - hop_length) // hop_length + 1
    labels = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_frames):
        start = i * hop_length
        end = start + hop_length
        frame = audio[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold:
            labels[i] = 1
    return labels


# ══════════════════════════════════════════════════════
# 2. MLP 模型
# ══════════════════════════════════════════════════════

class SpeakerMLP(nn.Module):
    def __init__(self, input_dim=45, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpeakerLSTM(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=128, n_layers=2, output_dim=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, feat) or (batch, feat)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F)
        out, _ = self.lstm(x)   # (B, T, H*2)
        out = self.dropout(out)
        out = self.fc(out)      # (B, T, 2)
        return out.squeeze(1)   # (B, 2) if single frame


class SpeakerCNN(nn.Module):
    def __init__(self, input_dim=45, output_dim=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: (B, 45)
        x = x.unsqueeze(1)  # (B, 1, 45)
        x = self.conv(x)     # (B, 128, 1)
        x = x.squeeze(-1)    # (B, 128)
        return self.fc(x)


# ══════════════════════════════════════════════════════
# 3. 訓練
# ══════════════════════════════════════════════════════

def prepare_data():
    """
    真正的 Speaker-Dependent 訓練資料：
    - Positive (1): hsuan 語音幀（高能量段）
    - Negative (0): 其他人 (0911, FEMH) 語音幀 + 所有靜音幀
    """
    # 收集所有說話者的音檔
    hsuan_files = sorted(MAT_DIR.glob("*hsuan*.wav"))
    interf_files = sorted(MAT_DIR.glob("*0911*.wav"))
    femh_files = sorted(MAT_DIR.glob("*FEMH*.wav"))

    all_features = []
    all_labels = []

    # hsuan: 語音幀 = 1, 靜音幀 = 0
    print(f"\n=== Positive class: hsuan ({len(hsuan_files)} files) ===")
    for f in hsuan_files:
        audio = read_audio(str(f))
        features = extract_sdmfcc(audio)
        vad = energy_vad_labels(audio)
        min_len = min(len(features), len(vad))
        features = features[:min_len]
        vad = vad[:min_len]

        # hsuan 語音幀 → label=1, hsuan 靜音幀 → label=0
        labels = vad.copy()  # 1=speech(hsuan), 0=silence

        all_features.append(features)
        all_labels.append(labels)

        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        print(f"  {f.name[:50]:52s} target={n_pos:4d} silence={n_neg:4d}")

    # 0911: 語音幀 = 0 (非目標說話者)
    print(f"\n=== Negative class: 0911 ({len(interf_files)} files) ===")
    for f in interf_files:
        audio = read_audio(str(f))
        features = extract_sdmfcc(audio)
        vad = energy_vad_labels(audio)
        min_len = min(len(features), len(vad))
        features = features[:min_len]
        vad = vad[:min_len]

        # 0911 語音幀 → label=0 (非目標), 靜音幀 → label=0
        labels = np.zeros(min_len, dtype=np.int64)

        all_features.append(features)
        all_labels.append(labels)

        n_speech = (vad == 1).sum()
        print(f"  {f.name[:50]:52s} non-target_speech={n_speech:4d} silence={min_len-n_speech:4d}")

    # FEMH: 語音幀 = 0 (非目標說話者)
    print(f"\n=== Negative class: FEMH ({len(femh_files)} files) ===")
    for f in femh_files:
        audio = read_audio(str(f))
        features = extract_sdmfcc(audio)
        vad = energy_vad_labels(audio)
        min_len = min(len(features), len(vad))
        features = features[:min_len]
        vad = vad[:min_len]

        labels = np.zeros(min_len, dtype=np.int64)

        all_features.append(features)
        all_labels.append(labels)

        n_speech = (vad == 1).sum()
        print(f"  {f.name[:50]:52s} non-target_speech={n_speech:4d} silence={min_len-n_speech:4d}")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\n總幀數: {len(X)}")
    print(f"  Target (hsuan speech): {(y==1).sum()} ({(y==1).mean():.1%})")
    print(f"  Non-target (others + silence): {(y==0).sum()} ({(y==0).mean():.1%})")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, y_train, y_val, mean, std


def train_model(model, X_train, X_val, y_train, y_val, epochs=50, lr=1e-3, name="model"):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total
        scheduler.step(1 - val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"    [{name}] Epoch {epoch+1:3d}: loss={total_loss/len(train_dl):.4f}, val_acc={val_acc:.3f} (best={best_acc:.3f})")

    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    [{name}] Best val acc: {best_acc:.3f}, params: {n_params:,}")
    return model, best_acc, n_params


def train_sklearn_models(X_train, y_train, X_val, y_val):
    """訓練 SVM 和 Random Forest"""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    results = {}

    # SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_val, svm.predict(X_val))
    print(f"    [SVM] val acc: {svm_acc:.3f}")
    results["SVM_SD"] = (svm, svm_acc)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"    [RF] val acc: {rf_acc:.3f}")
    results["RF_SD"] = (rf, rf_acc)

    return results


# ══════════════════════════════════════════════════════
# 4. 測試場景
# ══════════════════════════════════════════════════════

def build_scenarios():
    hsuan_files = sorted(MAT_DIR.glob("*hsuan_7.wav"))
    interf_files = sorted(MAT_DIR.glob("*0911*_7.wav"))
    femh_files = sorted(MAT_DIR.glob("*FEMH_7.wav"))

    hsuan_audios = [read_audio(str(f)) for f in hsuan_files]
    interf_audios = [read_audio(str(f)) for f in interf_files]
    femh_audio = read_audio(str(femh_files[0])) if femh_files else None

    seg_len = int(2.5 * SAMPLE_RATE)
    get_seg = lambda audios, idx: audios[idx % len(audios)][:seg_len]

    scenario_a = np.concatenate([
        get_seg(hsuan_audios, 0), get_seg(interf_audios, 0),
        get_seg(hsuan_audios, 1), get_seg(interf_audios, 1),
        get_seg(hsuan_audios, 2),
    ])
    a_labels = [
        ("hsuan_0.0-2.5s", 0.0, 2.5, True),
        ("interferer_2.5-5.0s", 2.5, 5.0, False),
        ("hsuan_5.0-7.5s", 5.0, 7.5, True),
        ("interferer_7.5-10.0s", 7.5, 10.0, False),
        ("hsuan_10.0-12.5s", 10.0, 12.5, True),
    ]

    overlap_len = int(3.0 * SAMPLE_RATE)
    overlap = hsuan_audios[0][seg_len:seg_len + overlap_len] + interf_audios[0][:overlap_len]
    scenario_b = np.concatenate([get_seg(hsuan_audios, 0), overlap, get_seg(hsuan_audios, 1)])
    b_labels = [
        ("hsuan_only_0.0-2.5s", 0.0, 2.5, True),
        ("hsuan+interferer_2.5-5.5s", 2.5, 5.5, True),
        ("hsuan_only_5.5-8.0s", 5.5, 8.0, True),
    ]

    noise = np.random.RandomState(42).randn(len(scenario_a)).astype(np.float32)
    noise *= np.sqrt(np.mean(scenario_a ** 2) / (10 ** (10 / 10)) / (np.mean(noise ** 2) + 1e-12))
    scenario_c = scenario_a + noise

    if femh_audio is not None:
        scenario_d = np.concatenate([
            get_seg(hsuan_audios, 0), femh_audio[:seg_len],
            get_seg(hsuan_audios, 1), femh_audio[seg_len:2 * seg_len],
            get_seg(hsuan_audios, 2),
        ])
    else:
        scenario_d = scenario_a.copy()
    d_labels = [
        ("hsuan_0.0-2.5s", 0.0, 2.5, True),
        ("interferer_2.5-5.0s", 2.5, 5.0, False),
        ("hsuan_5.0-7.5s", 5.0, 7.5, True),
        ("interferer_7.5-10.0s", 7.5, 10.0, False),
        ("hsuan_10.0-12.5s", 10.0, 12.5, True),
    ]

    return {
        "scenario_a": (scenario_a, a_labels),
        "scenario_b": (scenario_b, b_labels),
        "scenario_c": (scenario_c, a_labels),
        "scenario_d": (scenario_d, d_labels),
    }


def evaluate(predictions, labels, hop=512):
    results = {}
    frame_dur = hop / SAMPLE_RATE
    for name, start_sec, end_sec, is_target in labels:
        sf = int(start_sec / frame_dur)
        ef = min(int(end_sec / frame_dur), len(predictions))
        seg = predictions[sf:ef]
        if len(seg) == 0:
            continue
        results[name] = {
            "target_ratio": round(float(np.mean(seg)), 4),
            "is_target": is_target,
            "n_frames": len(seg),
        }
    return results


def predict_torch(model, features):
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(features))
        return out.argmax(dim=1).numpy()


def predict_sklearn(model, features):
    return model.predict(features)


def compute_f1(all_results, scenarios):
    tp = fn = fp = tn = 0
    for sc_name in ["scenario_a", "scenario_c", "scenario_d"]:
        for seg_name, seg_data in all_results.get(sc_name, {}).items():
            n = seg_data["n_frames"]
            t = int(seg_data["target_ratio"] * n)
            nt = n - t
            if seg_data["is_target"]:
                tp += t; fn += nt
            else:
                fp += t; tn += nt
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main():
    print("=" * 70)
    print("Speaker-Dependent 多架構比較實驗")
    print("只用 hsuan 語音訓練，測試四場景")
    print("=" * 70)

    # 準備資料
    X_train, X_val, y_train, y_val, mean, std = prepare_data()

    # 訓練所有模型
    print("\n" + "=" * 70)
    print("訓練模型...")
    print("=" * 70)

    models = {}

    # PyTorch 模型
    mlp = SpeakerMLP(45, [256, 128, 64], 2, 0.3)
    mlp, mlp_acc, mlp_params = train_model(mlp, X_train, X_val, y_train, y_val, epochs=50, name="MLP")
    models["MLP_SD"] = {"model": mlp, "type": "torch", "acc": mlp_acc, "params": mlp_params}

    lstm = SpeakerLSTM(45, 128, 2, 2, 0.3)
    lstm, lstm_acc, lstm_params = train_model(lstm, X_train, X_val, y_train, y_val, epochs=50, name="LSTM")
    models["LSTM_SD"] = {"model": lstm, "type": "torch", "acc": lstm_acc, "params": lstm_params}

    cnn = SpeakerCNN(45, 2, 0.3)
    cnn, cnn_acc, cnn_params = train_model(cnn, X_train, X_val, y_train, y_val, epochs=50, name="CNN")
    models["CNN_SD"] = {"model": cnn, "type": "torch", "acc": cnn_acc, "params": cnn_params}

    # Sklearn 模型
    sk_results = train_sklearn_models(X_train, y_train, X_val, y_val)
    for name, (sk_model, sk_acc) in sk_results.items():
        models[name] = {"model": sk_model, "type": "sklearn", "acc": sk_acc, "params": "N/A"}

    # 測試場景
    print("\n" + "=" * 70)
    print("四場景測試")
    print("=" * 70)

    scenarios = build_scenarios()
    pvad_report = PROJECT_DIR / "test_parallel" / "test_report_parallel.json"
    pvad_metrics = {}
    if pvad_report.exists():
        with open(pvad_report, encoding="utf-8") as f:
            pvad_metrics = json.load(f).get("metrics", {})

    all_model_results = {}

    for model_name, model_info in models.items():
        all_model_results[model_name] = {}
        for sc_name, (audio, labels) in scenarios.items():
            features = extract_sdmfcc(audio)
            features_norm = (features - mean) / (std + 1e-8)

            if model_info["type"] == "torch":
                preds = predict_torch(model_info["model"], features_norm)
            else:
                preds = predict_sklearn(model_info["model"], features_norm)

            all_model_results[model_name][sc_name] = evaluate(preds, labels)

    # 打印比較表
    print(f"\n{'=' * 110}")
    print(f"COMPARISON TABLE (all Speaker-Dependent + pVAD-SE)")
    print(f"{'=' * 110}")

    model_names = list(models.keys())
    header = f"{'Segment':32s}"
    for mn in model_names:
        header += f" {mn:>8s}"
    header += f" {'pVAD-SE':>8s} {'GT':>4s}"
    print(header)
    print("-" * len(header))

    for sc_name in ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]:
        print(f"\n  [{sc_name.upper()}]")
        # Get segment names from first model
        first_model = model_names[0]
        pvad_sc = pvad_metrics.get(sc_name, {})

        for seg_name in all_model_results[first_model].get(sc_name, {}):
            gt_data = all_model_results[first_model][sc_name][seg_name]
            gt = "T" if gt_data["is_target"] else "I"
            line = f"  {seg_name:32s}"
            for mn in model_names:
                tr = all_model_results[mn].get(sc_name, {}).get(seg_name, {}).get("target_ratio", 0)
                line += f" {tr:7.1%}"
            pvad_tr = pvad_sc.get(seg_name, {}).get("target_ratio", -1)
            pvad_s = f"{pvad_tr:.1%}" if pvad_tr >= 0 else "N/A"
            line += f" {pvad_s:>8s} {gt:>4s}"
            print(line)

    # F1 比較
    print(f"\n{'=' * 110}")
    print(f"F1 SCORE SUMMARY (場景 A+C+D)")
    print(f"{'=' * 110}")
    print(f"{'Model':>12s} {'Val Acc':>8s} {'Params':>10s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
    print("-" * 72)

    all_f1 = {}
    for model_name in model_names:
        metrics = compute_f1(all_model_results[model_name], scenarios)
        all_f1[model_name] = metrics
        val_acc = models[model_name]["acc"]
        params = models[model_name]["params"]
        params_s = f"{params:,}" if isinstance(params, int) else params
        print(f"{model_name:>12s} {val_acc:7.1%} {params_s:>10s} {metrics['accuracy']:9.1%} {metrics['precision']:10.1%} {metrics['recall']:7.1%} {metrics['f1']:7.3f}")

    print(f"{'pVAD-SE':>12s} {'N/A':>8s} {'6.9M+83K':>10s} {'65.0%':>10s} {'76.8%':>10s} {'59.8%':>8s} {'0.672':>8s}")

    # 儲存
    output = {
        "experiment": "Speaker-Dependent multi-architecture comparison",
        "training_data": "hsuan only (MAT dataset, 8 files)",
        "features": "45-dim SDMFCC (15 MFCC + 15 Delta + 15 Delta2)",
        "models": {},
    }
    for model_name in model_names:
        metrics = all_f1[model_name]
        output["models"][model_name] = {
            "val_accuracy": round(models[model_name]["acc"], 4),
            "params": models[model_name]["params"],
            "test_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            "per_scenario": all_model_results[model_name],
        }

    out_path = PROJECT_DIR / "test_parallel" / "speaker_dependent_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n報告已存: {out_path}")


if __name__ == "__main__":
    main()
