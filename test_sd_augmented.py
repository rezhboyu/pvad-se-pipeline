#!/usr/bin/env python3
"""
Speaker-Dependent + 噪音增強訓練實驗
測試噪音增強是否能讓 SDMFCC 模型抗噪
"""
import sys, json, numpy as np, librosa
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.audio import read_audio, SAMPLE_RATE

MAT_DIR = Path.home() / "Desktop" / "VOICE" / "MAT"

def extract_sdmfcc(audio, sr=16000, hop_length=512):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T

def energy_vad(audio, hop=512, thr=0.01):
    n = (len(audio) - hop) // hop + 1
    return np.array([1 if np.sqrt(np.mean(audio[i*hop:i*hop+hop]**2)) > thr else 0 for i in range(n)], dtype=np.int64)

def add_noise(audio, snr_db):
    noise = np.random.randn(len(audio)).astype(np.float32)
    sp = np.mean(audio**2)
    np_ = sp / (10 ** (snr_db / 10))
    return audio + noise * np.sqrt(np_ / (np.mean(noise**2) + 1e-12))

# ══════ 資料準備 ══════
hsuan_files = sorted(MAT_DIR.glob("*hsuan*.wav"))
interf_files = sorted(MAT_DIR.glob("*0911*.wav"))
femh_files = sorted(MAT_DIR.glob("*FEMH*.wav"))

all_X, all_y = [], []
snr_levels = [5, 10, 15, 20]

print("=== 資料準備 (原始 + 噪音增強) ===")
for f in hsuan_files:
    a = read_audio(str(f))
    feat = extract_sdmfcc(a); vad = energy_vad(a)
    ml = min(len(feat), len(vad))
    all_X.append(feat[:ml]); all_y.append(vad[:ml])
    for snr in snr_levels:
        fn = add_noise(a, snr)
        feat_n = extract_sdmfcc(fn)
        ml_n = min(len(feat_n), len(vad))
        all_X.append(feat_n[:ml_n]); all_y.append(vad[:ml_n])

for f in list(interf_files) + list(femh_files):
    a = read_audio(str(f))
    feat = extract_sdmfcc(a); ml = len(feat)
    all_X.append(feat); all_y.append(np.zeros(ml, dtype=np.int64))
    for snr in snr_levels:
        fn = add_noise(a, snr)
        feat_n = extract_sdmfcc(fn)
        all_X.append(feat_n); all_y.append(np.zeros(len(feat_n), dtype=np.int64))

X = np.concatenate(all_X); y = np.concatenate(all_y)
print(f"增強後: {len(X)} frames, Target={int((y==1).sum())} ({(y==1).mean():.1%})")

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mean = X_tr.mean(0); std = X_tr.std(0) + 1e-8
X_tr_n = (X_tr - mean) / std; X_val_n = (X_val - mean) / std

# ══════ 訓練 ══════
print("\n=== 訓練模型 ===")
models = {}

svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
svm.fit(X_tr_n, y_tr)
print(f"SVM_aug val acc: {accuracy_score(y_val, svm.predict(X_val_n)):.3f}")
models["SVM_aug"] = svm

mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=200, random_state=42)
mlp.fit(X_tr_n, y_tr)
print(f"MLP_aug val acc: {accuracy_score(y_val, mlp.predict(X_val_n)):.3f}")
models["MLP_aug"] = mlp

rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42)
rf.fit(X_tr_n, y_tr)
print(f"RF_aug val acc: {accuracy_score(y_val, rf.predict(X_val_n)):.3f}")
models["RF_aug"] = rf

# ══════ 測試場景 ══════
seg_len = int(2.5 * SAMPLE_RATE)
ha = [read_audio(str(f)) for f in hsuan_files[:3]]
ia = [read_audio(str(f)) for f in interf_files[:2]]
fa = read_audio(str(femh_files[0]))
gs = lambda a, i: a[i % len(a)][:seg_len]

scenarios = {}
scenarios["scenario_a"] = np.concatenate([gs(ha,0), gs(ia,0), gs(ha,1), gs(ia,1), gs(ha,2)])
scenarios["scenario_d"] = np.concatenate([gs(ha,0), fa[:seg_len], gs(ha,1), fa[seg_len:2*seg_len], gs(ha,2)])
noise = np.random.RandomState(42).randn(len(scenarios["scenario_a"])).astype(np.float32)
noise *= np.sqrt(np.mean(scenarios["scenario_a"]**2) / 10 / (np.mean(noise**2) + 1e-12))
scenarios["scenario_c"] = scenarios["scenario_a"] + noise

seg_labels = [
    ("hsuan_0.0-2.5s", 0, 2.5, True), ("interferer_2.5-5.0s", 2.5, 5, False),
    ("hsuan_5.0-7.5s", 5, 7.5, True), ("interferer_7.5-10.0s", 7.5, 10, False),
    ("hsuan_10.0-12.5s", 10, 12.5, True),
]

# Load originals for comparison
with open("test_parallel/speaker_dependent_comparison.json", encoding="utf-8") as f:
    sd_orig = json.load(f)
with open("test_parallel/test_report_parallel.json", encoding="utf-8") as f:
    pvad_m = json.load(f)["metrics"]

hop = 512; fd = hop / SAMPLE_RATE

print("\n" + "=" * 100)
print("比較: 原始 SD vs 增強 SD vs pVAD-SE")
print("=" * 100)
header = f"{'Segment':>28s} {'SVM_orig':>9s} {'SVM_aug':>9s} {'MLP_orig':>9s} {'MLP_aug':>9s} {'RF_aug':>9s} {'pVAD-SE':>9s} {'GT':>4s}"
print(header)
print("-" * len(header))

for sc in ["scenario_a", "scenario_c", "scenario_d"]:
    print(f"\n  [{sc.upper()}]")
    audio = scenarios[sc]
    feat = (extract_sdmfcc(audio) - mean) / std

    preds = {mn: m.predict(feat) for mn, m in models.items()}

    for sn, ss, se, is_t in seg_labels:
        sf = int(ss / fd); ef = min(int(se / fd), len(feat))
        gt = "T" if is_t else "I"

        vals = [f"{np.mean(preds[mn][sf:ef]):.1%}" for mn in models]

        # Original SD
        svm_o = sd_orig["models"].get("SVM_SD", {}).get("per_scenario", {}).get(sc, {}).get(sn, {}).get("target_ratio", -1)
        mlp_o = sd_orig["models"].get("MLP_SD", {}).get("per_scenario", {}).get(sc, {}).get(sn, {}).get("target_ratio", -1)
        pvad = pvad_m.get(sc, {}).get(sn, {}).get("target_ratio", -1)

        so = f"{svm_o:.1%}" if svm_o >= 0 else "N/A"
        mo = f"{mlp_o:.1%}" if mlp_o >= 0 else "N/A"
        ps = f"{pvad:.1%}" if pvad >= 0 else "N/A"

        print(f"  {sn:>28s} {so:>9s} {vals[0]:>9s} {mo:>9s} {vals[1]:>9s} {vals[2]:>9s} {ps:>9s} {gt:>4s}")

# F1 summary
print("\n" + "=" * 80)
print("F1 Score 比較")
print("=" * 80)
print(f"{'Model':>12s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
print("-" * 52)

for mn, m in models.items():
    tp = fn = fp = tn = 0
    for sc in ["scenario_a", "scenario_c", "scenario_d"]:
        audio = scenarios[sc]
        feat = (extract_sdmfcc(audio) - mean) / std
        p = m.predict(feat)
        for sn, ss, se, is_t in seg_labels:
            sf = int(ss/fd); ef = min(int(se/fd), len(p))
            t = int(np.sum(p[sf:ef])); nt = ef - sf - t
            if is_t: tp += t; fn += nt
            else: fp += t; tn += nt
    pr = tp/(tp+fp) if (tp+fp) else 0
    rc = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc) else 0
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f"{mn:>12s} {acc:9.1%} {pr:10.1%} {rc:7.1%} {f1:7.3f}")

print(f"{'SVM_orig':>12s} {'72.8%':>10s} {'77.4%':>10s} {'77.2%':>8s} {'0.773':>8s}")
print(f"{'MLP_orig':>12s} {'70.3%':>10s} {'76.3%':>10s} {'73.4%':>8s} {'0.748':>8s}")
print(f"{'pVAD-SE':>12s} {'65.0%':>10s} {'76.8%':>10s} {'59.8%':>8s} {'0.672':>8s}")
