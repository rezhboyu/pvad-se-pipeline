#!/usr/bin/env python3
"""
pVAD Baseline 分類器比較實驗
============================
比較不同分類架構作為 pVAD 說話者驗證的 baseline：
- Cosine Similarity（threshold-based）
- MLP（多層感知機）
- SVM（RBF / Linear kernel）
- Logistic Regression
- KNN（K-Nearest Neighbors）
- PLDA（簡化版，LDA + cosine）

目標說話者：hsuan
非目標說話者：0911636193、FEMH

評估方式：Leave-One-Out Cross-Validation（因資料量極小）
指標：Accuracy、Precision、Recall、F1、EER
視覺化：ROC 曲線比較

用法：
    python baseline_comparison.py
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

# MAT 音頻資料夾（三位說話者的測試音頻）
# 在 VM 中使用掛載路徑，本機使用 ~/Desktop/VOICE/MAT
_mat_vm = Path("/sessions/festive-dazzling-brahmagupta/mnt/MAT")
_mat_local = Path(os.path.expanduser("~/Desktop/VOICE/MAT"))
MAT_DIR = _mat_vm if _mat_vm.exists() else _mat_local
# 輸出目錄
OUTPUT_DIR = PROJECT_DIR / "baseline_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 匯入 pipeline 的模組 ──────────────────────────────
from utils.audio import read_audio, SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder

# ── 匯入 ML 工具 ──────────────────────────────────────
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # 無頭模式，不需要 GUI
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 1. 提取所有音頻的 Speaker Embeddings
# ══════════════════════════════════════════════════════
def extract_all_embeddings(mat_dir: Path, encoder: SpeakerEncoder):
    """
    掃描 MAT 資料夾，提取每個 wav 檔的全句級 speaker embedding。

    回傳：
        embeddings: list of np.ndarray (每個 shape = (256,))
        labels: list of int (1 = 目標說話者 hsuan, 0 = 非目標)
        speaker_names: list of str (說話者名稱)
        file_names: list of str (檔案名稱)
    """
    wav_files = sorted(mat_dir.glob("*.wav"))

    embeddings = []
    labels = []
    speaker_names = []
    file_names = []

    print(f"\n{'='*60}")
    print(f"提取 Speaker Embeddings")
    print(f"{'='*60}")
    print(f"音頻資料夾: {mat_dir}")
    print(f"找到 {len(wav_files)} 個 wav 檔\n")

    for wav_path in wav_files:
        # 從檔名解析說話者名稱
        # 格式: uuid_speakerName_channel.wav
        parts = wav_path.stem.split("_")
        # UUID 含有 '-'，說話者名稱在中間，最後是 channel
        # 需要找出說話者名稱：去掉第一個 UUID 部分和最後一個 channel 部分
        # UUID 格式固定為 8-4-4-4-12，所以整個 UUID 是第一個部分
        # 但由於 '_' 分割，UUID 本身沒有 '_'，所以 parts[0] 就是 UUID
        # 不對，UUID 有 '-' 沒有 '_'，所以 split('_') 後：
        # parts[0] = full UUID, parts[1] = speaker name, parts[2] = channel
        # 但要注意有些說話者名稱本身可能不含 '_'

        # 簡單方式：用最後一個 '_' 分割出 channel，再用第一個 UUID 後分割出 speaker
        stem = wav_path.stem
        # 找到第一個 UUID 結束的位置（UUID 長度固定 36 字元）
        uuid_part = stem[:36]
        rest = stem[37:]  # 跳過 UUID 後的 '_'
        # rest 格式: speakerName_channel
        last_underscore = rest.rfind("_")
        speaker = rest[:last_underscore]
        channel = rest[last_underscore+1:]

        # 判定標籤：hsuan = 目標（1），其他 = 非目標（0）
        label = 1 if speaker == "hsuan" else 0

        # 提取 embedding
        audio = read_audio(str(wav_path))
        embedding = encoder.extract_embedding(audio)

        embeddings.append(embedding)
        labels.append(label)
        speaker_names.append(speaker)
        file_names.append(wav_path.name)

        duration = len(audio) / SAMPLE_RATE
        print(f"  [{len(embeddings):2d}] {speaker:15s} (ch{channel}) "
              f"| {duration:5.1f}s | label={label} | {wav_path.name[:20]}...")

    print(f"\n總計: {len(embeddings)} 個 embedding")
    print(f"  目標 (hsuan): {sum(labels)} 個")
    print(f"  非目標:       {len(labels) - sum(labels)} 個")

    return (np.array(embeddings), np.array(labels),
            speaker_names, file_names)


# ══════════════════════════════════════════════════════
# 2. 建構特徵矩陣
# ══════════════════════════════════════════════════════
def build_pair_features(embeddings, labels, mode="concat"):
    """
    建構 pair-based 特徵矩陣。

    兩種模式：
    - "concat": 拼接 enrollment 均值 embedding 和測試 embedding → (512,)
    - "diff": 差值 + 元素乘積 → (512,)

    由於資料太少，我們直接用 enrollment 均值作為 reference，
    對每個測試樣本建立一個特徵向量。

    但這裡更合理的做法是：直接用 embedding 做分類，
    label 就是 是否為目標說話者。
    """
    pass  # 見下方的 build_verification_features


def build_verification_features(embeddings, labels):
    """
    建構說話者驗證的特徵矩陣。

    策略：對每個樣本，計算它與所有 hsuan 樣本的 cosine similarity 統計量。
    但由於 LOO，我們不能用測試樣本自己，所以需要在 LOO 循環中動態建構。

    簡單版：直接用 raw embedding 做分類。
    """
    return embeddings, labels


# ══════════════════════════════════════════════════════
# 3. 計算 EER（Equal Error Rate）
# ══════════════════════════════════════════════════════
def compute_eer(y_true, y_scores):
    """
    計算 Equal Error Rate。
    EER 是 FAR = FRR 的交叉點。

    Parameters
    ----------
    y_true : array-like, 真實標籤 (0/1)
    y_scores : array-like, 預測分數（越高越可能是正類）

    Returns
    -------
    eer : float, Equal Error Rate
    eer_threshold : float, EER 對應的閾值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # 找到 FAR ≈ FRR 的點
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx] if idx < len(thresholds) else 0

    return eer, eer_threshold


# ══════════════════════════════════════════════════════
# 4. 定義各分類器
# ══════════════════════════════════════════════════════
def get_classifiers():
    """
    回傳所有要比較的分類器。

    每個分類器都包在 Pipeline 裡（含 StandardScaler），
    因為 SVM / MLP 等對特徵尺度敏感。

    注意：由於資料量極小（~13 個樣本），
    複雜模型幾乎必然過擬合或欠擬合，但仍有比較價值。
    """
    classifiers = {}

    # 1. MLP（多層感知機）— 小型網路
    classifiers["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),  # 兩層，夠小以避免嚴重過擬合
            activation='relu',
            max_iter=2000,
            random_state=42,
            early_stopping=False,  # 資料太少不做 early stopping
            alpha=0.01,           # L2 正規化，防止過擬合
        ))
    ])

    # 2. SVM (RBF kernel)
    classifiers["SVM-RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,    # 需要概率輸出來畫 ROC
            random_state=42,
        ))
    ])

    # 3. SVM (Linear kernel)
    classifiers["SVM-Linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=42,
        ))
    ])

    # 4. Logistic Regression
    classifiers["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ))
    ])

    # 5. KNN
    # K 不能超過訓練集大小，LOO 下訓練集 = N-1 = 12
    classifiers["KNN (K=3)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=3,
            metric='cosine',  # 用 cosine 距離，更適合 embedding 空間
        ))
    ])

    # 6. PLDA 簡化版：LDA 降維 + cosine
    # 注意：真正的 PLDA 需要 within-class / between-class 協方差估計，
    # 在只有 3 個說話者的情況下幾乎不可行。
    # 這裡用 LDA 做降維再分類作為替代。
    classifiers["LDA (PLDA替代)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(
            solver='svd',  # SVD solver 不需要 covariance 計算
        ))
    ])

    return classifiers


# ══════════════════════════════════════════════════════
# 5. Cosine Similarity Baseline
# ══════════════════════════════════════════════════════
def evaluate_cosine_baseline(embeddings, labels, thresholds=None):
    """
    用 LOO 評估 cosine similarity baseline。

    策略：
    - 對每個測試樣本，用剩餘的 hsuan 樣本計算 enrollment 均值
    - 計算測試樣本與 enrollment 均值的 cosine similarity
    - 用閾值判定是否為目標說話者

    回傳最佳閾值下的指標和所有分數。
    """
    n = len(labels)
    loo = LeaveOneOut()

    scores = np.zeros(n)

    for train_idx, test_idx in loo.split(embeddings):
        test_i = test_idx[0]
        train_embs = embeddings[train_idx]
        train_labels = labels[train_idx]

        # enrollment: 訓練集中 hsuan 樣本的均值
        hsuan_mask = train_labels == 1
        if hsuan_mask.sum() == 0:
            scores[test_i] = 0.0
            continue

        enrollment = train_embs[hsuan_mask].mean(axis=0)
        # L2 正規化
        enrollment = enrollment / (np.linalg.norm(enrollment) + 1e-8)

        # cosine similarity（embedding 已經 L2 正規化過）
        test_emb = embeddings[test_i]
        test_emb = test_emb / (np.linalg.norm(test_emb) + 1e-8)
        scores[test_i] = float(np.dot(enrollment, test_emb))

    # 找最佳閾值
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.01)

    best_f1 = -1
    best_thresh = 0.5
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # 用最佳閾值計算指標
    preds = (scores >= best_thresh).astype(int)

    return {
        "scores": scores,
        "predictions": preds,
        "threshold": best_thresh,
    }


# ══════════════════════════════════════════════════════
# 6. LOO 評估所有有監督分類器
# ══════════════════════════════════════════════════════
def evaluate_classifiers_loo(embeddings, labels, classifiers):
    """
    用 Leave-One-Out 評估所有分類器。

    回傳每個分類器的預測結果和概率分數。
    """
    n = len(labels)
    loo = LeaveOneOut()
    results = {}

    for name, clf in classifiers.items():
        print(f"\n  評估 {name}...")
        predictions = np.zeros(n, dtype=int)
        scores = np.zeros(n)

        success_count = 0
        fail_count = 0

        for train_idx, test_idx in loo.split(embeddings):
            test_i = test_idx[0]
            X_train = embeddings[train_idx]
            y_train = labels[train_idx]
            X_test = embeddings[test_idx]

            try:
                # LDA 在二分類中只有 1 個判別軸，
                # 且需要每個類別至少有一個樣本
                if len(np.unique(y_train)) < 2:
                    predictions[test_i] = 0
                    scores[test_i] = 0.0
                    fail_count += 1
                    continue

                clf_copy = clf  # sklearn Pipeline 支援 fit 覆蓋
                clf.fit(X_train, y_train)
                predictions[test_i] = clf.predict(X_test)[0]

                # 取得概率分數（用於 ROC 曲線）
                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(X_test)
                    # 正類（hsuan = 1）的概率
                    if proba.shape[1] == 2:
                        scores[test_i] = proba[0, 1]
                    else:
                        scores[test_i] = proba[0, 0]
                elif hasattr(clf, 'decision_function'):
                    scores[test_i] = clf.decision_function(X_test)[0]
                else:
                    scores[test_i] = float(predictions[test_i])

                success_count += 1

            except Exception as e:
                print(f"    ⚠️ fold {test_i} 失敗: {e}")
                predictions[test_i] = 0
                scores[test_i] = 0.0
                fail_count += 1

        results[name] = {
            "predictions": predictions,
            "scores": scores,
            "success": success_count,
            "fail": fail_count,
        }

        print(f"    完成: {success_count} 成功, {fail_count} 失敗")

    return results


# ══════════════════════════════════════════════════════
# 7. 計算所有指標
# ══════════════════════════════════════════════════════
def compute_all_metrics(labels, results_dict):
    """
    計算所有分類器的評估指標。

    回傳格式：
    {
        "method_name": {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "eer": float,
            "auc": float,
        }
    }
    """
    metrics = {}

    for name, result in results_dict.items():
        preds = result["predictions"]
        scores = result["scores"]

        m = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

        # EER 和 AUC（需要概率分數）
        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            m["auc"] = auc(fpr, tpr)
            eer_val, eer_thresh = compute_eer(labels, scores)
            m["eer"] = eer_val
            m["eer_threshold"] = eer_thresh
        except Exception:
            m["auc"] = 0.0
            m["eer"] = 1.0
            m["eer_threshold"] = 0.0

        metrics[name] = m

    return metrics


# ══════════════════════════════════════════════════════
# 8. 繪製 ROC 曲線
# ══════════════════════════════════════════════════════
def plot_roc_curves(labels, results_dict, output_path):
    """
    繪製所有分類器的 ROC 曲線比較圖。
    """
    plt.figure(figsize=(10, 8))

    # 配色方案
    colors = [
        '#e74c3c',  # 紅
        '#3498db',  # 藍
        '#2ecc71',  # 綠
        '#f39c12',  # 橙
        '#9b59b6',  # 紫
        '#1abc9c',  # 青
        '#e67e22',  # 深橙
    ]

    for idx, (name, result) in enumerate(results_dict.items()):
        scores = result["scores"]
        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            color = colors[idx % len(colors)]
            plt.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name} (AUC={roc_auc:.3f})')
        except Exception as e:
            print(f"  ⚠️ 無法繪製 {name} 的 ROC: {e}")

    # 對角線（隨機分類器）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.500)')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curves: pVAD Baseline Classifier Comparison\n'
              '(Target Speaker: hsuan | Leave-One-Out CV)', fontsize=13)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ ROC 曲線已儲存: {output_path}")


# ══════════════════════════════════════════════════════
# 9. 繪製指標比較柱狀圖
# ══════════════════════════════════════════════════════
def plot_metrics_comparison(metrics, output_path):
    """
    繪製各分類器的指標比較柱狀圖。
    """
    methods = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    display_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    n_methods = len(methods)
    n_metrics = len(metric_names)
    x = np.arange(n_methods)
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for i, (metric, display, color) in enumerate(zip(metric_names, display_names, colors)):
        values = [metrics[m].get(metric, 0) for m in methods]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=display, color=color, alpha=0.85)

        # 在柱子上方標註數值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Classification Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('pVAD Baseline: Performance Metrics Comparison\n'
                 '(Target: hsuan | LOO-CV | 13 samples)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 指標比較圖已儲存: {output_path}")


# ══════════════════════════════════════════════════════
# 10. 繪製 Embedding 分布視覺化 (t-SNE / PCA)
# ══════════════════════════════════════════════════════
def plot_embedding_distribution(embeddings, labels, speaker_names, output_path):
    """
    用 PCA 降維後視覺化 embedding 分布。
    （資料太少不適合 t-SNE，用 PCA 較穩定）
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 按說話者分組繪製
    unique_speakers = sorted(set(speaker_names))
    colors = {'hsuan': '#e74c3c', '0911636193': '#3498db', 'FEMH': '#2ecc71'}
    markers = {'hsuan': 'o', '0911636193': 's', 'FEMH': '^'}

    for speaker in unique_speakers:
        mask = [s == speaker for s in speaker_names]
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                  c=colors.get(speaker, '#999999'),
                  marker=markers.get(speaker, 'o'),
                  s=100, label=f'{speaker} ({"target" if speaker == "hsuan" else "non-target"})',
                  edgecolors='black', linewidths=0.5, alpha=0.8)

    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)', fontsize=11)
    ax.set_title('Speaker Embedding Distribution (PCA)\n'
                 'WeSpeaker ResNet34-LM, 256-dim', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Embedding 分布圖已儲存: {output_path}")


# ══════════════════════════════════════════════════════
# 11. 繪製 Cosine Similarity 分布圖
# ══════════════════════════════════════════════════════
def plot_similarity_distribution(embeddings, labels, output_path):
    """
    繪製目標/非目標說話者的 cosine similarity 分布圖。
    """
    # 計算所有樣本與 hsuan enrollment 均值的 cosine similarity
    hsuan_mask = labels == 1
    enrollment = embeddings[hsuan_mask].mean(axis=0)
    enrollment = enrollment / (np.linalg.norm(enrollment) + 1e-8)

    target_sims = []
    nontarget_sims = []

    for i in range(len(labels)):
        emb = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
        sim = float(np.dot(enrollment, emb))
        if labels[i] == 1:
            target_sims.append(sim)
        else:
            nontarget_sims.append(sim)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 1, 20)
    ax.hist(target_sims, bins=bins, alpha=0.7, color='#e74c3c',
           label=f'Target (hsuan, n={len(target_sims)})', edgecolor='black')
    ax.hist(nontarget_sims, bins=bins, alpha=0.7, color='#3498db',
           label=f'Non-target (n={len(nontarget_sims)})', edgecolor='black')

    ax.axvline(x=0.25, color='green', linestyle='--', linewidth=2,
              label='Pipeline threshold (0.25)')

    ax.set_xlabel('Cosine Similarity', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Cosine Similarity Distribution\n'
                 '(vs. hsuan enrollment mean)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Similarity 分布圖已儲存: {output_path}")


# ══════════════════════════════════════════════════════
# 12. 生成文字報告
# ══════════════════════════════════════════════════════
def generate_report(metrics, embeddings, labels, speaker_names,
                    cosine_result, output_path):
    """
    生成 Markdown 格式的比較報告。
    """
    lines = []
    lines.append("# pVAD Baseline 分類器比較報告")
    lines.append("")
    lines.append("## 1. 實驗設定")
    lines.append("")
    lines.append(f"- **目標說話者**: hsuan")
    lines.append(f"- **非目標說話者**: 0911636193, FEMH")
    lines.append(f"- **總樣本數**: {len(labels)}")
    lines.append(f"- **目標樣本數**: {sum(labels)} ({sum(labels)/len(labels):.1%})")
    lines.append(f"- **非目標樣本數**: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels):.1%})")
    lines.append(f"- **Embedding 模型**: WeSpeaker ResNet34-LM (256-dim)")
    lines.append(f"- **評估方式**: Leave-One-Out Cross-Validation")
    lines.append("")

    # 各說話者樣本分布
    lines.append("### 說話者樣本分布")
    lines.append("")
    speaker_counts = defaultdict(int)
    for s in speaker_names:
        speaker_counts[s] += 1
    lines.append("| 說話者 | 樣本數 | 類別 |")
    lines.append("|--------|--------|------|")
    for speaker, count in sorted(speaker_counts.items()):
        cat = "目標" if speaker == "hsuan" else "非目標"
        lines.append(f"| {speaker} | {count} | {cat} |")
    lines.append("")

    # 指標比較表
    lines.append("## 2. 指標比較")
    lines.append("")
    lines.append("| Method | Accuracy | Precision | Recall | F1 | AUC | EER |")
    lines.append("|--------|----------|-----------|--------|----|-----|-----|")

    # 按 F1 排序
    sorted_methods = sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    for name, m in sorted_methods:
        lines.append(
            f"| {name} "
            f"| {m['accuracy']:.3f} "
            f"| {m['precision']:.3f} "
            f"| {m['recall']:.3f} "
            f"| {m['f1']:.3f} "
            f"| {m['auc']:.3f} "
            f"| {m['eer']:.3f} |"
        )
    lines.append("")

    # Cosine baseline 的最佳閾值
    lines.append(f"**Cosine Similarity 最佳閾值**: {cosine_result['threshold']:.2f}")
    lines.append("")

    # 分析
    lines.append("## 3. 分析")
    lines.append("")

    best_method = sorted_methods[0][0]
    best_f1 = sorted_methods[0][1]['f1']
    cosine_f1 = metrics.get("Cosine Similarity", {}).get('f1', 0)

    lines.append(f"### 最佳方法: {best_method} (F1={best_f1:.3f})")
    lines.append("")

    lines.append("### 關鍵觀察")
    lines.append("")
    lines.append("1. **資料量限制**: 總共只有 13 個樣本，這對任何有監督學習方法都是極大的挑戰。"
                "LOO-CV 在此情況下是最合理的評估策略，但結果的統計顯著性有限。")
    lines.append("")
    lines.append("2. **Cosine Similarity baseline**: 作為最簡單的方法，cosine similarity "
                "不需要訓練，完全依賴 embedding 空間的幾何結構。在資料極少的情況下，"
                "這反而可能是最穩定的方法。")
    lines.append("")
    lines.append("3. **有監督方法**: MLP、SVM 等方法在如此小的資料集上容易過擬合或欠擬合。"
                "它們的表現很大程度上取決於 embedding 空間是否已經把不同說話者分得很開。")
    lines.append("")
    lines.append("4. **實用建議**: 在資料量增加之前，cosine similarity 或簡單的 "
                "threshold-based 方法可能是最實際的選擇。當累積更多說話者和樣本後，"
                "可以重新評估有監督方法的優勢。")
    lines.append("")

    lines.append("## 4. 輸出檔案")
    lines.append("")
    lines.append("- `roc_curves.png`: ROC 曲線比較圖")
    lines.append("- `metrics_comparison.png`: 指標柱狀圖")
    lines.append("- `embedding_pca.png`: Embedding PCA 分布圖")
    lines.append("- `similarity_distribution.png`: Cosine Similarity 分布圖")
    lines.append("- `embeddings.npz`: 所有提取的 speaker embeddings")
    lines.append("- `results.json`: 完整的數值結果")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  ✅ 報告已儲存: {output_path}")
    return report


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print(" pVAD Baseline 分類器比較實驗")
    print("=" * 60)

    # ── 1. 載入 Speaker Encoder ───────────────────────
    print("\n[1/7] 載入 WeSpeaker ResNet34-LM...")
    model_path = PROJECT_DIR / "models" / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    assert model_path.exists(), f"模型不存在: {model_path}"
    encoder = SpeakerEncoder(str(model_path))
    print(f"  ✅ 模型載入完成 (embedding dim: {encoder.embed_dim})")

    # ── 2. 提取所有 Embeddings ────────────────────────
    print("\n[2/7] 提取所有音頻的 Speaker Embeddings...")
    embeddings, labels, speaker_names, file_names = extract_all_embeddings(
        MAT_DIR, encoder
    )

    # 儲存 embeddings
    np.savez(
        OUTPUT_DIR / "embeddings.npz",
        embeddings=embeddings,
        labels=labels,
        speaker_names=speaker_names,
        file_names=file_names,
    )
    print(f"  ✅ Embeddings 已儲存: {OUTPUT_DIR / 'embeddings.npz'}")

    # ── 3. 評估 Cosine Similarity Baseline ────────────
    print("\n[3/7] 評估 Cosine Similarity Baseline...")
    cosine_result = evaluate_cosine_baseline(embeddings, labels)
    print(f"  最佳閾值: {cosine_result['threshold']:.2f}")

    # ── 4. 評估所有有監督分類器 ───────────────────────
    print("\n[4/7] 評估有監督分類器 (LOO-CV)...")
    classifiers = get_classifiers()
    clf_results = evaluate_classifiers_loo(embeddings, labels, classifiers)

    # 合併所有結果
    all_results = {"Cosine Similarity": cosine_result}
    all_results.update(clf_results)

    # ── 5. 計算指標 ──────────────────────────────────
    print("\n[5/7] 計算評估指標...")
    metrics = compute_all_metrics(labels, all_results)

    # 輸出指標表格
    print(f"\n{'Method':<22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s} {'AUC':>6s} {'EER':>6s}")
    print("-" * 70)
    for name, m in sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<22s} {m['accuracy']:6.3f} {m['precision']:6.3f} "
              f"{m['recall']:6.3f} {m['f1']:6.3f} {m['auc']:6.3f} {m['eer']:6.3f}")

    # ── 6. 繪製圖表 ──────────────────────────────────
    print("\n[6/7] 繪製圖表...")
    plot_roc_curves(labels, all_results, OUTPUT_DIR / "roc_curves.png")
    plot_metrics_comparison(metrics, OUTPUT_DIR / "metrics_comparison.png")
    plot_embedding_distribution(embeddings, labels, speaker_names,
                               OUTPUT_DIR / "embedding_pca.png")
    plot_similarity_distribution(embeddings, labels,
                                OUTPUT_DIR / "similarity_distribution.png")

    # ── 7. 生成報告 ──────────────────────────────────
    print("\n[7/7] 生成報告...")
    report = generate_report(
        metrics, embeddings, labels, speaker_names,
        cosine_result, OUTPUT_DIR / "report.md"
    )

    # 儲存完整數值結果為 JSON
    json_results = {}
    for name, m in metrics.items():
        json_results[name] = {k: float(v) for k, v in m.items()}

    with open(OUTPUT_DIR / "results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 數值結果已儲存: {OUTPUT_DIR / 'results.json'}")

    # ── 完成 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" 實驗完成！所有結果已輸出到:")
    print(f" {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
