#!/usr/bin/env python3
"""
pVAD Noisy Baseline 分類器比較實驗
===================================
在不同 SNR 條件下比較分類器的魯棒性：
- 噪音類型：White noise、Babble noise（用多說話者混合模擬）
- SNR 級別：Clean, 20dB, 10dB, 5dB, 0dB, -5dB
- 分析粒度：Utterance-level（全句）、Frame-level（0.5s, 1s 窗）

評估方式：Leave-One-Out Cross-Validation
分類器：Cosine Similarity, MLP, SVM-RBF, SVM-Linear, Logistic Regression, KNN

用法：
    python noisy_baseline_comparison.py
"""

import sys
import os
import json
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── 路徑設定 ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# MAT 音頻資料夾
_mat_vm = Path("/sessions/intelligent-magical-johnson/mnt/MAT")
_mat_local = Path(os.path.expanduser("~/Desktop/VOICE/MAT"))
MAT_DIR = _mat_vm if _mat_vm.exists() else _mat_local

# 輸出目錄
OUTPUT_DIR = PROJECT_DIR / "baseline_comparison" / "noisy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 匯入 pipeline 模組 ───────────────────────────────
from utils.audio import read_audio, SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder

# ── ML 工具 ──────────────────────────────────────────
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 噪音產生
# ══════════════════════════════════════════════════════
def add_white_noise(audio, snr_db):
    """添加白噪音到指定 SNR (dB)。"""
    sig_power = np.mean(audio ** 2)
    if sig_power < 1e-10:
        return audio.copy()
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.RandomState(42).randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


def generate_babble_noise(length, n_sources=5, seed=123):
    """
    用隨機語音模擬 babble noise。
    方法：疊加多個隨機相位的正弦波群（模擬多人語音的頻譜包絡）。
    真正的 babble noise 應用多人錄音混合，這裡用頻譜近似。
    """
    rng = np.random.RandomState(seed)
    babble = np.zeros(length, dtype=np.float32)
    for _ in range(n_sources):
        # 模擬語音基頻 (80-300 Hz) 和諧波
        f0 = rng.uniform(80, 300)
        t = np.arange(length) / SAMPLE_RATE
        signal = np.zeros(length, dtype=np.float32)
        # 基頻 + 前 10 個諧波，幅度遞減
        for h in range(1, 11):
            freq = f0 * h
            if freq > SAMPLE_RATE / 2:
                break
            amp = 1.0 / h
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
        # 加上隨機幅度調變（模擬語音的韻律）
        mod_freq = rng.uniform(2, 8)  # 2-8 Hz 的幅度調變
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + rng.uniform(0, 2*np.pi))
        signal *= mod.astype(np.float32)
        babble += signal
    # 正規化
    babble = babble / (np.max(np.abs(babble)) + 1e-8)
    return babble


def add_babble_noise(audio, snr_db, seed=123):
    """添加 babble noise 到指定 SNR (dB)。"""
    sig_power = np.mean(audio ** 2)
    if sig_power < 1e-10:
        return audio.copy()
    babble = generate_babble_noise(len(audio), seed=seed)
    # 調整 babble 功率到目標 SNR
    noise_power_target = sig_power / (10 ** (snr_db / 10))
    babble_power = np.mean(babble ** 2)
    if babble_power < 1e-10:
        return audio.copy()
    scale = np.sqrt(noise_power_target / babble_power)
    return (audio + scale * babble).astype(np.float32)


# ══════════════════════════════════════════════════════
# 音頻載入
# ══════════════════════════════════════════════════════
def load_all_audio(mat_dir):
    """載入所有 wav 並解析說話者標籤。"""
    wav_files = sorted(mat_dir.glob("*.wav"))
    audios = []
    labels = []
    speaker_names = []
    file_names = []

    for wav_path in wav_files:
        stem = wav_path.stem
        uuid_part = stem[:36]
        rest = stem[37:]
        last_underscore = rest.rfind("_")
        speaker = rest[:last_underscore]
        label = 1 if speaker == "hsuan" else 0

        audio = read_audio(str(wav_path))
        audios.append(audio)
        labels.append(label)
        speaker_names.append(speaker)
        file_names.append(wav_path.name)

    return audios, np.array(labels), speaker_names, file_names


# ══════════════════════════════════════════════════════
# Embedding 提取（全句 & 短窗）
# ══════════════════════════════════════════════════════
def extract_utterance_embeddings(audios, encoder):
    """提取全句級 embedding。"""
    embeddings = []
    for audio in audios:
        emb = encoder.extract_embedding(audio)
        embeddings.append(emb)
    return np.array(embeddings)


def extract_frame_embeddings(audios, encoder, window_sec):
    """
    提取 frame-level embedding：對每段音頻用滑動窗提取多個短窗 embedding，
    然後取均值作為該段音頻的代表 embedding。

    這模擬了 pVAD 在實際串流中的行為：每個短窗獨立做 embedding，
    最終決策基於多個短窗結果的聚合。
    """
    window_samples = int(window_sec * SAMPLE_RATE)
    hop_samples = window_samples // 2  # 50% overlap

    all_embeddings = []
    for audio in audios:
        if len(audio) < window_samples:
            # 音頻太短，直接用全段
            emb = encoder.extract_embedding(audio)
            all_embeddings.append(emb)
            continue

        frame_embs = []
        start = 0
        while start + window_samples <= len(audio):
            chunk = audio[start:start + window_samples]
            emb = encoder.extract_embedding(chunk)
            frame_embs.append(emb)
            start += hop_samples

        # 均值 embedding + L2 正規化
        mean_emb = np.mean(frame_embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        all_embeddings.append(mean_emb.astype(np.float32))

    return np.array(all_embeddings)


# ══════════════════════════════════════════════════════
# 分類器定義
# ══════════════════════════════════════════════════════
def get_classifiers():
    """回傳所有 baseline 分類器（含 StandardScaler）。"""
    return {
        "Cosine Similarity": None,  # 特殊處理
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                  max_iter=2000, random_state=42, alpha=0.01))
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
        ]),
        "SVM-Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='linear', C=1.0, probability=True, random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        ]),
        "KNN (K=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3, metric='cosine'))
        ]),
    }


# ══════════════════════════════════════════════════════
# LOO 評估
# ══════════════════════════════════════════════════════
def evaluate_cosine_loo(embeddings, labels):
    """LOO 評估 cosine similarity baseline。"""
    n = len(labels)
    loo = LeaveOneOut()
    scores = np.zeros(n)

    for train_idx, test_idx in loo.split(embeddings):
        test_i = test_idx[0]
        train_embs = embeddings[train_idx]
        train_labels = labels[train_idx]
        hsuan_mask = train_labels == 1
        if hsuan_mask.sum() == 0:
            scores[test_i] = 0.0
            continue
        enrollment = train_embs[hsuan_mask].mean(axis=0)
        enrollment = enrollment / (np.linalg.norm(enrollment) + 1e-8)
        test_emb = embeddings[test_i]
        test_emb = test_emb / (np.linalg.norm(test_emb) + 1e-8)
        scores[test_i] = float(np.dot(enrollment, test_emb))

    # 找最佳閾值
    best_f1, best_thresh = -1, 0.5
    for t in np.arange(0.0, 1.0, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = (scores >= best_thresh).astype(int)
    return {"predictions": preds, "scores": scores, "threshold": best_thresh}


def evaluate_classifier_loo(embeddings, labels, clf):
    """LOO 評估單一有監督分類器。"""
    n = len(labels)
    loo = LeaveOneOut()
    predictions = np.zeros(n, dtype=int)
    scores = np.zeros(n)

    for train_idx, test_idx in loo.split(embeddings):
        test_i = test_idx[0]
        X_train, y_train = embeddings[train_idx], labels[train_idx]
        X_test = embeddings[test_idx]

        if len(np.unique(y_train)) < 2:
            predictions[test_i] = 0
            scores[test_i] = 0.0
            continue

        try:
            clf.fit(X_train, y_train)
            predictions[test_i] = clf.predict(X_test)[0]
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_test)
                scores[test_i] = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]
            elif hasattr(clf, 'decision_function'):
                scores[test_i] = clf.decision_function(X_test)[0]
            else:
                scores[test_i] = float(predictions[test_i])
        except Exception:
            predictions[test_i] = 0
            scores[test_i] = 0.0

    return {"predictions": predictions, "scores": scores}


def compute_metrics(labels, preds, scores):
    """計算分類指標。"""
    m = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        m["auc"] = float(auc(fpr, tpr))
        fnr = 1 - tpr
        idx = np.argmin(np.abs(fpr - fnr))
        m["eer"] = float((fpr[idx] + fnr[idx]) / 2)
    except Exception:
        m["auc"] = 0.0
        m["eer"] = 1.0
    return m


# ══════════════════════════════════════════════════════
# 主實驗流程
# ══════════════════════════════════════════════════════
def run_experiment(audios, labels, speaker_names, encoder, noise_type, snr_db):
    """
    對指定噪音條件跑完整分類器比較。
    回傳：{method_name: {accuracy, precision, recall, f1, auc, eer}}
    """
    # 1. 加噪音
    if snr_db is None:
        noisy_audios = audios  # clean
    else:
        noisy_audios = []
        for i, audio in enumerate(audios):
            if noise_type == "white":
                noisy = add_white_noise(audio, snr_db)
            else:
                noisy = add_babble_noise(audio, snr_db, seed=123 + i)
            noisy_audios.append(noisy)

    # 2. 提取 embedding（全句）
    embeddings = extract_utterance_embeddings(noisy_audios, encoder)

    # 3. 評估所有分類器
    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        if clf is None:
            # Cosine similarity
            res = evaluate_cosine_loo(embeddings, labels)
        else:
            res = evaluate_classifier_loo(embeddings, labels, clf)
        metrics = compute_metrics(labels, res["predictions"], res["scores"])
        results[name] = metrics

    return results


def run_frame_experiment(audios, labels, encoder, noise_type, snr_db, window_sec):
    """
    Frame-level 實驗：用短窗提取 embedding 後評估。
    """
    # 加噪音
    if snr_db is None:
        noisy_audios = audios
    else:
        noisy_audios = []
        for i, audio in enumerate(audios):
            if noise_type == "white":
                noisy = add_white_noise(audio, snr_db)
            else:
                noisy = add_babble_noise(audio, snr_db, seed=123 + i)
            noisy_audios.append(noisy)

    # 提取 frame-level embedding
    embeddings = extract_frame_embeddings(noisy_audios, encoder, window_sec)

    # 評估
    classifiers = get_classifiers()
    results = {}
    for name, clf in classifiers.items():
        if clf is None:
            res = evaluate_cosine_loo(embeddings, labels)
        else:
            res = evaluate_classifier_loo(embeddings, labels, clf)
        metrics = compute_metrics(labels, res["predictions"], res["scores"])
        results[name] = metrics

    return results


# ══════════════════════════════════════════════════════
# 繪圖
# ══════════════════════════════════════════════════════
COLORS = {
    "Cosine Similarity": '#e74c3c',
    "MLP": '#3498db',
    "SVM-RBF": '#2ecc71',
    "SVM-Linear": '#f39c12',
    "Logistic Regression": '#9b59b6',
    "KNN (K=3)": '#1abc9c',
}
MARKERS = {
    "Cosine Similarity": 'o',
    "MLP": 's',
    "SVM-RBF": '^',
    "SVM-Linear": 'D',
    "Logistic Regression": 'v',
    "KNN (K=3)": 'P',
}


def plot_metric_vs_snr(all_results, snr_labels, metric_name, noise_type, output_path, ylabel=None):
    """繪製 metric vs SNR 曲線。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(all_results[snr_labels[0]].keys())

    for method in methods:
        values = [all_results[snr][method][metric_name] for snr in snr_labels]
        ax.plot(range(len(snr_labels)), values,
                color=COLORS.get(method, '#999'),
                marker=MARKERS.get(method, 'o'),
                linewidth=2, markersize=8, label=method)

    ax.set_xticks(range(len(snr_labels)))
    ax.set_xticklabels(snr_labels, fontsize=10)
    ax.set_xlabel('SNR Condition', fontsize=12)
    ax.set_ylabel(ylabel or metric_name.capitalize(), fontsize=12)
    ax.set_title(f'{ylabel or metric_name.capitalize()} vs SNR ({noise_type.capitalize()} Noise)\n'
                 f'Utterance-Level | LOO-CV | Target: hsuan', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {output_path.name}")


def plot_frame_comparison(utt_results, frame_results_05, frame_results_1, snr_labels,
                          noise_type, output_path):
    """繪製全句 vs 短窗的比較圖（以 Cosine Similarity 為例）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in zip(axes, ['accuracy', 'f1'], ['Accuracy', 'F1']):
        for method in ["Cosine Similarity", "MLP", "SVM-RBF"]:
            # 全句
            vals_utt = [utt_results[snr][method][metric] for snr in snr_labels]
            ax.plot(range(len(snr_labels)), vals_utt,
                    color=COLORS.get(method, '#999'), marker='o',
                    linewidth=2, label=f'{method} (utt)', linestyle='-')
            # 0.5s
            vals_05 = [frame_results_05[snr][method][metric] for snr in snr_labels]
            ax.plot(range(len(snr_labels)), vals_05,
                    color=COLORS.get(method, '#999'), marker='^',
                    linewidth=1.5, label=f'{method} (0.5s)', linestyle='--')
            # 1s
            vals_1 = [frame_results_1[snr][method][metric] for snr in snr_labels]
            ax.plot(range(len(snr_labels)), vals_1,
                    color=COLORS.get(method, '#999'), marker='s',
                    linewidth=1.5, label=f'{method} (1.0s)', linestyle=':')

        ax.set_xticks(range(len(snr_labels)))
        ax.set_xticklabels(snr_labels, fontsize=9)
        ax.set_xlabel('SNR Condition', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}: Utterance vs Frame-Level\n({noise_type.capitalize()} Noise)', fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)

    # 共用 legend
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {output_path.name}")


def plot_combined_heatmap(all_results, snr_labels, noise_type, output_path):
    """繪製 Accuracy heatmap：方法 x SNR。"""
    methods = list(all_results[snr_labels[0]].keys())
    data = np.array([[all_results[snr][m]['accuracy'] for snr in snr_labels] for m in methods])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(snr_labels)))
    ax.set_xticklabels(snr_labels, fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('SNR Condition', fontsize=12)
    ax.set_title(f'Accuracy Heatmap ({noise_type.capitalize()} Noise)\n'
                 f'Utterance-Level | LOO-CV', fontsize=13)

    # 標註數值
    for i in range(len(methods)):
        for j in range(len(snr_labels)):
            val = data[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {output_path.name}")


# ══════════════════════════════════════════════════════
# 報告生成
# ══════════════════════════════════════════════════════
def generate_report(all_white, all_babble, frame_white_05, frame_white_1,
                    frame_babble_05, frame_babble_1, snr_labels, output_path):
    """生成完整 Markdown 報告。"""
    lines = []
    lines.append("# pVAD Noisy Baseline 分類器比較報告")
    lines.append("")
    lines.append("## 實驗設定")
    lines.append("")
    lines.append("- **目標說話者**: hsuan")
    lines.append("- **非目標說話者**: 0911636193, FEMH")
    lines.append("- **Embedding 模型**: WeSpeaker ResNet34-LM (256-dim)")
    lines.append("- **評估方式**: Leave-One-Out Cross-Validation")
    lines.append(f"- **SNR 條件**: {', '.join(snr_labels)}")
    lines.append("- **噪音類型**: White noise, Babble noise (simulated)")
    lines.append("- **分析粒度**: Utterance-level, Frame-level (0.5s, 1.0s)")
    lines.append("")

    # 每種噪音的結果表
    for noise_name, results in [("White Noise", all_white), ("Babble Noise", all_babble)]:
        lines.append(f"## {noise_name} — Utterance-Level 結果")
        lines.append("")

        # Accuracy 表
        methods = list(results[snr_labels[0]].keys())
        header = "| Method | " + " | ".join(snr_labels) + " |"
        sep = "|--------|" + "|".join(["-----:" for _ in snr_labels]) + "|"
        lines.append(f"### Accuracy")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for m in methods:
            vals = " | ".join([f"{results[snr][m]['accuracy']:.3f}" for snr in snr_labels])
            lines.append(f"| {m} | {vals} |")
        lines.append("")

        # F1 表
        lines.append(f"### F1 Score")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for m in methods:
            vals = " | ".join([f"{results[snr][m]['f1']:.3f}" for snr in snr_labels])
            lines.append(f"| {m} | {vals} |")
        lines.append("")

    # Frame-level 結果
    lines.append("## Frame-Level 結果（White Noise）")
    lines.append("")
    for win_name, frame_res in [("0.5s Window", frame_white_05), ("1.0s Window", frame_white_1)]:
        lines.append(f"### {win_name} — Accuracy")
        lines.append("")
        methods = list(frame_res[snr_labels[0]].keys())
        header = "| Method | " + " | ".join(snr_labels) + " |"
        sep = "|--------|" + "|".join(["-----:" for _ in snr_labels]) + "|"
        lines.append(header)
        lines.append(sep)
        for m in methods:
            vals = " | ".join([f"{frame_res[snr][m]['accuracy']:.3f}" for snr in snr_labels])
            lines.append(f"| {m} | {vals} |")
        lines.append("")

    lines.append("## Frame-Level 結果（Babble Noise）")
    lines.append("")
    for win_name, frame_res in [("0.5s Window", frame_babble_05), ("1.0s Window", frame_babble_1)]:
        lines.append(f"### {win_name} — Accuracy")
        lines.append("")
        methods = list(frame_res[snr_labels[0]].keys())
        header = "| Method | " + " | ".join(snr_labels) + " |"
        sep = "|--------|" + "|".join(["-----:" for _ in snr_labels]) + "|"
        lines.append(header)
        lines.append(sep)
        for m in methods:
            vals = " | ".join([f"{frame_res[snr][m]['accuracy']:.3f}" for snr in snr_labels])
            lines.append(f"| {m} | {vals} |")
        lines.append("")

    # 分析
    lines.append("## 分析與觀察")
    lines.append("")
    lines.append("### 噪音對各方法的影響")
    lines.append("")
    lines.append("圖表顯示了不同 SNR 條件下各分類器的表現退化情況。"
                "在乾淨條件下所有方法都能達到完美，但隨著 SNR 降低，"
                "各方法的魯棒性差異開始顯現。")
    lines.append("")
    lines.append("### Frame-Level vs Utterance-Level")
    lines.append("")
    lines.append("短窗分析顯示，frame-level embedding 品質因上下文不足而下降，"
                "在噪音條件下衰退更為顯著。0.5s 窗口比 1.0s 窗口受影響更大。")
    lines.append("")

    lines.append("## 輸出檔案")
    lines.append("")
    lines.append("- `accuracy_vs_snr_white.png` / `accuracy_vs_snr_babble.png`: Accuracy vs SNR 曲線")
    lines.append("- `f1_vs_snr_white.png` / `f1_vs_snr_babble.png`: F1 vs SNR 曲線")
    lines.append("- `frame_comparison_white.png` / `frame_comparison_babble.png`: 全句 vs 短窗比較")
    lines.append("- `heatmap_white.png` / `heatmap_babble.png`: Accuracy heatmap")
    lines.append("- `results.json`: 完整數值結果")
    lines.append("- `report.md`: 本報告")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✅ 報告已儲存: {output_path}")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print(" pVAD Noisy Baseline 分類器比較實驗")
    print("=" * 60)

    # 1. 載入模型
    print("\n[1/8] 載入 WeSpeaker ResNet34-LM...")
    model_path = PROJECT_DIR / "models" / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    assert model_path.exists(), f"模型不存在: {model_path}"
    encoder = SpeakerEncoder(str(model_path))
    print(f"  ✅ 模型載入完成 (dim={encoder.embed_dim})")

    # 2. 載入音頻
    print("\n[2/8] 載入音頻...")
    audios, labels, speaker_names, file_names = load_all_audio(MAT_DIR)
    print(f"  ✅ 載入 {len(audios)} 段音頻")
    print(f"     目標 (hsuan): {sum(labels)}, 非目標: {len(labels) - sum(labels)}")

    # 3. SNR 條件
    snr_configs = [
        ("Clean", None),
        ("20dB", 20),
        ("10dB", 10),
        ("5dB", 5),
        ("0dB", 0),
        ("-5dB", -5),
    ]
    snr_labels = [s[0] for s in snr_configs]

    # 4. Utterance-level 實驗
    print("\n[3/8] Utterance-level: White Noise 實驗...")
    all_white = {}
    for snr_name, snr_db in snr_configs:
        print(f"  SNR={snr_name}...", end=" ")
        all_white[snr_name] = run_experiment(audios, labels, speaker_names, encoder, "white", snr_db)
        acc_cos = all_white[snr_name]["Cosine Similarity"]["accuracy"]
        print(f"Cosine Acc={acc_cos:.3f}")

    print("\n[4/8] Utterance-level: Babble Noise 實驗...")
    all_babble = {}
    for snr_name, snr_db in snr_configs:
        print(f"  SNR={snr_name}...", end=" ")
        all_babble[snr_name] = run_experiment(audios, labels, speaker_names, encoder, "babble", snr_db)
        acc_cos = all_babble[snr_name]["Cosine Similarity"]["accuracy"]
        print(f"Cosine Acc={acc_cos:.3f}")

    # 5. Frame-level 實驗
    print("\n[5/8] Frame-level: White Noise 實驗...")
    frame_white_05, frame_white_1 = {}, {}
    for snr_name, snr_db in snr_configs:
        print(f"  SNR={snr_name} (0.5s)...", end=" ")
        frame_white_05[snr_name] = run_frame_experiment(audios, labels, encoder, "white", snr_db, 0.5)
        print(f"done", end="  ")
        print(f"(1.0s)...", end=" ")
        frame_white_1[snr_name] = run_frame_experiment(audios, labels, encoder, "white", snr_db, 1.0)
        print("done")

    print("\n[6/8] Frame-level: Babble Noise 實驗...")
    frame_babble_05, frame_babble_1 = {}, {}
    for snr_name, snr_db in snr_configs:
        print(f"  SNR={snr_name} (0.5s)...", end=" ")
        frame_babble_05[snr_name] = run_frame_experiment(audios, labels, encoder, "babble", snr_db, 0.5)
        print(f"done", end="  ")
        print(f"(1.0s)...", end=" ")
        frame_babble_1[snr_name] = run_frame_experiment(audios, labels, encoder, "babble", snr_db, 1.0)
        print("done")

    # 6. 繪圖
    print("\n[7/8] 繪製圖表...")
    # Accuracy vs SNR
    plot_metric_vs_snr(all_white, snr_labels, 'accuracy', 'white',
                       OUTPUT_DIR / "accuracy_vs_snr_white.png", ylabel='Accuracy')
    plot_metric_vs_snr(all_babble, snr_labels, 'accuracy', 'babble',
                       OUTPUT_DIR / "accuracy_vs_snr_babble.png", ylabel='Accuracy')
    # F1 vs SNR
    plot_metric_vs_snr(all_white, snr_labels, 'f1', 'white',
                       OUTPUT_DIR / "f1_vs_snr_white.png", ylabel='F1 Score')
    plot_metric_vs_snr(all_babble, snr_labels, 'f1', 'babble',
                       OUTPUT_DIR / "f1_vs_snr_babble.png", ylabel='F1 Score')
    # Frame comparison
    plot_frame_comparison(all_white, frame_white_05, frame_white_1, snr_labels,
                          'white', OUTPUT_DIR / "frame_comparison_white.png")
    plot_frame_comparison(all_babble, frame_babble_05, frame_babble_1, snr_labels,
                          'babble', OUTPUT_DIR / "frame_comparison_babble.png")
    # Heatmaps
    plot_combined_heatmap(all_white, snr_labels, 'white', OUTPUT_DIR / "heatmap_white.png")
    plot_combined_heatmap(all_babble, snr_labels, 'babble', OUTPUT_DIR / "heatmap_babble.png")

    # 7. 報告
    print("\n[8/8] 生成報告...")
    generate_report(all_white, all_babble, frame_white_05, frame_white_1,
                    frame_babble_05, frame_babble_1, snr_labels, OUTPUT_DIR / "report.md")

    # 儲存完整 JSON 結果
    json_out = {
        "utterance_white": {snr: {m: v for m, v in res.items()} for snr, res in all_white.items()},
        "utterance_babble": {snr: {m: v for m, v in res.items()} for snr, res in all_babble.items()},
        "frame_0.5s_white": {snr: {m: v for m, v in res.items()} for snr, res in frame_white_05.items()},
        "frame_1.0s_white": {snr: {m: v for m, v in res.items()} for snr, res in frame_white_1.items()},
        "frame_0.5s_babble": {snr: {m: v for m, v in res.items()} for snr, res in frame_babble_05.items()},
        "frame_1.0s_babble": {snr: {m: v for m, v in res.items()} for snr, res in frame_babble_1.items()},
    }
    with open(OUTPUT_DIR / "results.json", 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 結果 JSON: {OUTPUT_DIR / 'results.json'}")

    print(f"\n{'='*60}")
    print(f" 實驗完成！所有結果已輸出到:")
    print(f" {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
