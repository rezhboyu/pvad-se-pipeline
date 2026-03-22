#!/usr/bin/env python3
"""
Pair-wise Speaker Verification Baseline 比較實驗
==============================================
用 tse_aishell 資料集的 speaker_embedding (192維) 做 pair-wise verification。
給一對 embedding，判斷是否同一人。

資料集：
  - train: 4508 樣本, 400 說話者
  - val:   492 樣本, 284 說話者

分類器：
  1. Cosine Similarity (threshold-based)
  2. Logistic Regression
  3. SVM-RBF
  4. SVM-Linear
  5. MLP (2層, hidden=64)
  6. KNN
  7. LDA + Cosine (PLDA-like)

評估指標：EER, AUC, minDCF, Accuracy/Precision/Recall/F1
額外實驗：特徵表示比較、訓練集大小影響
"""

import json
import os
import sys
import time
import warnings
import itertools
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==================== 資料載入 ====================

def load_embeddings(data_dir):
    """載入某個 split 的所有 embedding 和 speaker_id"""
    samples = []
    for sample_dir in sorted(os.listdir(data_dir)):
        meta_path = os.path.join(data_dir, sample_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        emb = np.array(meta['speaker_embedding'], dtype=np.float32)
        spk = meta['speaker_id']
        samples.append({'embedding': emb, 'speaker_id': spk, 'sample_id': sample_dir})
    return samples


def build_speaker_dict(samples):
    """按 speaker_id 分組"""
    spk_dict = defaultdict(list)
    for s in samples:
        spk_dict[s['speaker_id']].append(s['embedding'])
    return spk_dict


def generate_pairs(samples, neg_ratio=1, max_pairs=None, seed=42):
    """
    產生 pair-wise 訓練/測試資料

    正對：同說話者的不同 utterance 組合
    負對：不同說話者的 utterance pair，按 neg_ratio 倍數取樣

    回傳：(emb_a_list, emb_b_list, labels)
    """
    rng = np.random.RandomState(seed)
    spk_dict = build_speaker_dict(samples)

    # 產生正對：同說話者的所有 utterance 兩兩組合
    pos_pairs = []
    for spk, embs in spk_dict.items():
        if len(embs) < 2:
            continue  # 只有 1 個 utterance 的說話者無法產生正對
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                pos_pairs.append((embs[i], embs[j]))

    n_pos = len(pos_pairs)
    n_neg_target = int(n_pos * neg_ratio)

    # 產生負對：隨機取不同說話者的 utterance
    all_spk_ids = list(spk_dict.keys())
    neg_pairs = []
    attempts = 0
    max_attempts = n_neg_target * 10
    while len(neg_pairs) < n_neg_target and attempts < max_attempts:
        spk_a, spk_b = rng.choice(all_spk_ids, size=2, replace=False)
        emb_a = spk_dict[spk_a][rng.randint(len(spk_dict[spk_a]))]
        emb_b = spk_dict[spk_b][rng.randint(len(spk_dict[spk_b]))]
        neg_pairs.append((emb_a, emb_b))
        attempts += 1

    # 合併並打亂
    all_pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    indices = list(range(len(all_pairs)))
    rng.shuffle(indices)

    emb_a = np.array([all_pairs[i][0] for i in indices])
    emb_b = np.array([all_pairs[i][1] for i in indices])
    y = np.array([labels[i] for i in indices])

    if max_pairs is not None and len(y) > max_pairs:
        emb_a = emb_a[:max_pairs]
        emb_b = emb_b[:max_pairs]
        y = y[:max_pairs]

    return emb_a, emb_b, y


# ==================== 特徵工程 ====================

def cosine_similarity_batch(a, b):
    """批次計算 cosine similarity"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.sum(a_norm * b_norm, axis=1)


def extract_features(emb_a, emb_b, mode='full'):
    """
    從 embedding pair 提取分類特徵

    mode:
      'cosine'  : 只用 cosine similarity (1維)
      'diff'    : |emb_a - emb_b| (192維)
      'product' : emb_a * emb_b (192維)
      'concat'  : [emb_a, emb_b] (384維)
      'diff_product' : [|diff|, product] (384維)
      'full'    : [|diff|, product, cosine] (385維)
    """
    cos = cosine_similarity_batch(emb_a, emb_b).reshape(-1, 1)
    diff = np.abs(emb_a - emb_b)
    prod = emb_a * emb_b

    if mode == 'cosine':
        return cos
    elif mode == 'diff':
        return diff
    elif mode == 'product':
        return prod
    elif mode == 'concat':
        return np.concatenate([emb_a, emb_b], axis=1)
    elif mode == 'diff_product':
        return np.concatenate([diff, prod], axis=1)
    elif mode == 'full':
        return np.concatenate([diff, prod, cos], axis=1)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")


# ==================== 評估指標 ====================

def compute_eer(y_true, scores):
    """計算 Equal Error Rate"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr

    # 找 FPR == FNR 的交叉點
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        # fallback: 找最近的點
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2

    return eer


def compute_min_dcf(y_true, scores, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """
    計算 minimum Detection Cost Function
    DCF = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = np.min(dcf)

    # 正規化
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf_norm = min_dcf / default_dcf

    return min_dcf_norm


def find_optimal_threshold(y_true, scores):
    """找讓 accuracy 最高的 threshold"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def evaluate_classifier(y_true, scores, name=""):
    """完整評估一個分類器"""
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                  precision_score, recall_score, f1_score,
                                  roc_curve)

    results = {}
    results['name'] = name
    results['eer'] = compute_eer(y_true, scores)
    results['auc'] = roc_auc_score(y_true, scores)
    results['min_dcf'] = compute_min_dcf(y_true, scores)

    # 在最佳 threshold 下的指標
    threshold = find_optimal_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    results['threshold'] = threshold
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # ROC 曲線資料（用於繪圖）
    fpr, tpr, _ = roc_curve(y_true, scores)
    results['fpr'] = fpr
    results['tpr'] = tpr

    # DET 曲線資料
    results['fnr'] = 1 - tpr

    return results


# ==================== 分類器定義 ====================

class CosineBaseline:
    """Cosine similarity threshold-based，無需訓練"""
    def __init__(self):
        self.name = "Cosine Similarity"

    def fit(self, X_train, y_train, emb_a_train=None, emb_b_train=None):
        pass  # 無需訓練

    def score(self, X_test, emb_a_test=None, emb_b_test=None):
        return cosine_similarity_batch(emb_a_test, emb_b_test)


class SupervisedClassifier:
    """包裝 sklearn 分類器，統一介面"""
    def __init__(self, clf, name, use_proba=True):
        self.clf = clf
        self.name = name
        self.use_proba = use_proba

    def fit(self, X_train, y_train, **kwargs):
        self.clf.fit(X_train, y_train)

    def score(self, X_test, **kwargs):
        if self.use_proba and hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X_test)[:, 1]
        else:
            return self.clf.decision_function(X_test)


class LDACosineClassifier:
    """PLDA-like: LDA 降維後算 cosine similarity"""
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.name = f"LDA+Cosine (dim={n_components})"
        self.lda = None

    def fit(self, X_train, y_train, emb_a_train=None, emb_b_train=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # 用原始 embedding + speaker label 做 LDA
        # 但這裡我們只有 pair label，所以用 pair 特徵做 LDA 降維
        n_comp = min(self.n_components, X_train.shape[1] - 1, 1)  # binary 最多 1 維
        self.lda = LinearDiscriminantAnalysis(n_components=n_comp)
        self.lda.fit(X_train, y_train)

    def score(self, X_test, **kwargs):
        # LDA transform 後取第一維作為 score
        return self.lda.transform(X_test)[:, 0]


def build_classifiers():
    """建立所有分類器"""
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.calibration import CalibratedClassifierCV

    classifiers = [
        CosineBaseline(),

        SupervisedClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
            "Logistic Regression"
        ),

        # 用 LinearSVC + decision_function，比 SVC(kernel='linear') 快很多
        SupervisedClassifier(
            LinearSVC(C=1.0, max_iter=2000, dual='auto'),
            "SVM-Linear",
            use_proba=False
        ),

        SupervisedClassifier(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            ),
            "MLP (64-32)"
        ),

        SupervisedClassifier(
            KNeighborsClassifier(n_neighbors=5, metric='cosine'),
            "KNN (k=5)"
        ),

        LDACosineClassifier(n_components=64),
    ]
    return classifiers


class SVMRBFClassifier:
    """SVM-RBF 單獨處理，用子採樣控制訓練時間"""
    def __init__(self, max_train=5000):
        from sklearn.svm import SVC
        self.clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        self.name = f"SVM-RBF (n≤{max_train})"
        self.max_train = max_train

    def fit(self, X_train, y_train, **kwargs):
        if len(y_train) > self.max_train:
            idx = np.random.RandomState(42).choice(len(y_train), self.max_train, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
        self.clf.fit(X_train, y_train)

    def score(self, X_test, **kwargs):
        return self.clf.decision_function(X_test)


# ==================== 繪圖 ====================

def plot_roc_curves(all_results, output_path):
    """繪製 ROC 曲線比較圖"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for res, color in zip(all_results, colors):
        label = f"{res['name']} (AUC={res['auc']:.4f}, EER={res['eer']:.4f})"
        ax.plot(res['fpr'], res['tpr'], color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Pair-wise Speaker Verification', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC 曲線已儲存: {output_path}")


def plot_det_curves(all_results, output_path):
    """繪製 DET 曲線（False Rejection Rate vs False Acceptance Rate）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for res, color in zip(all_results, colors):
        # DET 用 probit scale
        fpr = res['fpr']
        fnr = res['fnr']

        # 過濾掉 0 和 1（probit 無法處理）
        mask = (fpr > 1e-6) & (fpr < 1 - 1e-6) & (fnr > 1e-6) & (fnr < 1 - 1e-6)
        fpr_valid = fpr[mask]
        fnr_valid = fnr[mask]

        if len(fpr_valid) > 0:
            label = f"{res['name']} (EER={res['eer']:.4f})"
            ax.plot(norm.ppf(fpr_valid), norm.ppf(fnr_valid), color=color, lw=2, label=label)

    # 設定刻度
    ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
    tick_labels = [f"{t*100:.1f}%" for t in ticks]
    tick_locs = [norm.ppf(t) for t in ticks]

    valid_ticks = [(loc, lab) for loc, lab, t in zip(tick_locs, tick_labels, ticks)
                   if np.isfinite(loc)]
    if valid_ticks:
        locs, labs = zip(*valid_ticks)
        ax.set_xticks(list(locs))
        ax.set_xticklabels(list(labs))
        ax.set_yticks(list(locs))
        ax.set_yticklabels(list(labs))

    ax.set_xlabel('False Acceptance Rate', fontsize=12)
    ax.set_ylabel('False Rejection Rate', fontsize=12)
    ax.set_title('DET Curves - Pair-wise Speaker Verification', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  DET 曲線已儲存: {output_path}")


def plot_feature_comparison(feat_results, output_path):
    """繪製不同特徵表示的比較圖"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左圖：EER 比較
    ax = axes[0]
    classifiers = list(feat_results.keys())
    feat_modes = list(feat_results[classifiers[0]].keys())

    x = np.arange(len(feat_modes))
    width = 0.8 / len(classifiers)

    for i, clf_name in enumerate(classifiers):
        eers = [feat_results[clf_name][fm]['eer'] for fm in feat_modes]
        offset = (i - len(classifiers)/2 + 0.5) * width
        ax.bar(x + offset, eers, width, label=clf_name, alpha=0.8)

    ax.set_xlabel('Feature Representation', fontsize=11)
    ax.set_ylabel('EER', fontsize=11)
    ax.set_title('EER by Feature Representation', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_modes, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 右圖：AUC 比較
    ax = axes[1]
    for i, clf_name in enumerate(classifiers):
        aucs = [feat_results[clf_name][fm]['auc'] for fm in feat_modes]
        offset = (i - len(classifiers)/2 + 0.5) * width
        ax.bar(x + offset, aucs, width, label=clf_name, alpha=0.8)

    ax.set_xlabel('Feature Representation', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('AUC by Feature Representation', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_modes, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  特徵比較圖已儲存: {output_path}")


def plot_trainsize_effect(size_results, output_path):
    """繪製訓練集大小對效能的影響"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fractions = sorted(size_results[list(size_results.keys())[0]].keys())

    # EER
    ax = axes[0]
    for clf_name in size_results:
        eers = [size_results[clf_name][f]['eer'] for f in fractions]
        ax.plot([f*100 for f in fractions], eers, 'o-', label=clf_name, lw=2, markersize=8)
    ax.set_xlabel('Training Data (%)', fontsize=11)
    ax.set_ylabel('EER', fontsize=11)
    ax.set_title('EER vs Training Data Size', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1]
    for clf_name in size_results:
        aucs = [size_results[clf_name][f]['auc'] for f in fractions]
        ax.plot([f*100 for f in fractions], aucs, 'o-', label=clf_name, lw=2, markersize=8)
    ax.set_xlabel('Training Data (%)', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('AUC vs Training Data Size', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  訓練集大小影響圖已儲存: {output_path}")


# ==================== 主實驗 ====================

def main():
    warnings.filterwarnings('ignore')

    # 路徑設定
    DATASET_BASE = '/sessions/jolly-awesome-bardeen/mnt/datasets_copy/tse_aishell'
    OUTPUT_DIR = '/sessions/jolly-awesome-bardeen/mnt/pvad-se-pipeline/baseline_comparison/aishell_v2'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Pair-wise Speaker Verification Baseline 比較實驗")
    print("=" * 70)

    # ---- 1. 載入資料 ----
    print("\n[1/6] 載入資料...")
    train_samples = load_embeddings(os.path.join(DATASET_BASE, 'train'))
    val_samples = load_embeddings(os.path.join(DATASET_BASE, 'val'))

    train_spk = build_speaker_dict(train_samples)
    val_spk = build_speaker_dict(val_samples)
    print(f"  Train: {len(train_samples)} 樣本, {len(train_spk)} 說話者")
    print(f"  Val:   {len(val_samples)} 樣本, {len(val_spk)} 說話者")

    # ---- 2. 產生 pair-wise 資料 ----
    print("\n[2/6] 產生 pair-wise 資料...")

    # 訓練集 pairs (1:1 正負比), 限制總量以加速 SVM
    emb_a_train, emb_b_train, y_train = generate_pairs(train_samples, neg_ratio=1, max_pairs=20000, seed=42)
    print(f"  Train pairs: {len(y_train)} (正={y_train.sum()}, 負={len(y_train)-y_train.sum()})")

    # 測試集 pairs (1:1)
    emb_a_val, emb_b_val, y_val = generate_pairs(val_samples, neg_ratio=1, seed=123)
    print(f"  Val pairs:   {len(y_val)} (正={y_val.sum()}, 負={len(y_val)-y_val.sum()})")

    # 也產生 1:3 的測試集
    emb_a_val3, emb_b_val3, y_val3 = generate_pairs(val_samples, neg_ratio=3, seed=456)
    print(f"  Val pairs (1:3): {len(y_val3)} (正={y_val3.sum()}, 負={len(y_val3)-y_val3.sum()})")

    # 預提取特徵 (full mode)
    print("\n  提取 pair 特徵 (mode='full': |diff|+product+cosine, 385維)...")
    X_train = extract_features(emb_a_train, emb_b_train, mode='full')
    X_val = extract_features(emb_a_val, emb_b_val, mode='full')
    X_val3 = extract_features(emb_a_val3, emb_b_val3, mode='full')
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_val3: {X_val3.shape}")

    # 特徵標準化（對有監督方法有幫助）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_val3_scaled = scaler.transform(X_val3)

    # ---- 3. 訓練並評估所有分類器 ----
    print("\n[3/6] 訓練並評估分類器...")
    classifiers = build_classifiers()
    # 加入 SVM-RBF（子採樣版）
    classifiers.append(SVMRBFClassifier(max_train=5000))

    all_results_1to1 = []
    all_results_1to3 = []

    for clf in classifiers:
        t0 = time.time()
        print(f"\n  --- {clf.name} ---")

        if isinstance(clf, CosineBaseline):
            clf.fit(None, None)
            scores_val = clf.score(None, emb_a_test=emb_a_val, emb_b_test=emb_b_val)
            scores_val3 = clf.score(None, emb_a_test=emb_a_val3, emb_b_test=emb_b_val3)
        else:
            clf.fit(X_train_scaled, y_train)
            scores_val = clf.score(X_val_scaled)
            scores_val3 = clf.score(X_val3_scaled)

        elapsed = time.time() - t0

        # 評估 1:1
        res = evaluate_classifier(y_val, scores_val, name=clf.name)
        res['train_time'] = elapsed
        all_results_1to1.append(res)

        # 評估 1:3
        res3 = evaluate_classifier(y_val3, scores_val3, name=clf.name)
        all_results_1to3.append(res3)

        print(f"    [1:1] EER={res['eer']:.4f}  AUC={res['auc']:.4f}  "
              f"minDCF={res['min_dcf']:.4f}  F1={res['f1']:.4f}  ({elapsed:.1f}s)")
        print(f"    [1:3] EER={res3['eer']:.4f}  AUC={res3['auc']:.4f}  "
              f"minDCF={res3['min_dcf']:.4f}  F1={res3['f1']:.4f}")

    # ---- 4. 繪圖 ----
    print("\n[4/6] 繪製圖表...")
    plot_roc_curves(all_results_1to1, os.path.join(OUTPUT_DIR, 'roc_curves_1to1.png'))
    plot_roc_curves(all_results_1to3, os.path.join(OUTPUT_DIR, 'roc_curves_1to3.png'))
    plot_det_curves(all_results_1to1, os.path.join(OUTPUT_DIR, 'det_curves_1to1.png'))
    plot_det_curves(all_results_1to3, os.path.join(OUTPUT_DIR, 'det_curves_1to3.png'))

    # ---- 5. 額外實驗：特徵表示比較 ----
    print("\n[5/6] 額外實驗：特徵表示比較...")

    feat_modes = ['cosine', 'diff', 'product', 'diff_product', 'full']
    # 選幾個代表性分類器做比較
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    feat_classifiers = {
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, C=1.0),
        'MLP (64-32)': lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                              early_stopping=True, random_state=42),
    }

    feat_results = {}
    for clf_name, clf_factory in feat_classifiers.items():
        feat_results[clf_name] = {}
        for fm in feat_modes:
            X_tr = extract_features(emb_a_train, emb_b_train, mode=fm)
            X_va = extract_features(emb_a_val, emb_b_val, mode=fm)

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)

            clf = clf_factory()
            clf.fit(X_tr_s, y_train)

            if hasattr(clf, 'predict_proba'):
                scores = clf.predict_proba(X_va_s)[:, 1]
            else:
                scores = clf.decision_function(X_va_s)

            res = evaluate_classifier(y_val, scores, name=f"{clf_name}_{fm}")
            feat_results[clf_name][fm] = res
            print(f"  {clf_name} + {fm:15s}: EER={res['eer']:.4f}  AUC={res['auc']:.4f}")

    plot_feature_comparison(feat_results, os.path.join(OUTPUT_DIR, 'feature_comparison.png'))

    # ---- 6. 額外實驗：訓練集大小影響 ----
    print("\n[6/6] 額外實驗：訓練集大小影響...")

    fractions = [0.1, 0.5, 1.0]
    from sklearn.svm import LinearSVC as _LinearSVC
    size_classifiers = {
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, C=1.0),
        'MLP (64-32)': lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                              early_stopping=True, random_state=42),
        'SVM-Linear': lambda: _LinearSVC(C=1.0, max_iter=2000, dual='auto'),
    }

    size_results = {name: {} for name in size_classifiers}

    for frac in fractions:
        n = int(len(y_train) * frac)
        X_sub = X_train_scaled[:n]
        y_sub = y_train[:n]

        for clf_name, clf_factory in size_classifiers.items():
            clf = clf_factory()
            clf.fit(X_sub, y_sub)

            if hasattr(clf, 'predict_proba'):
                scores = clf.predict_proba(X_val_scaled)[:, 1]
            else:
                scores = clf.decision_function(X_val_scaled)

            res = evaluate_classifier(y_val, scores, name=f"{clf_name}_{frac}")
            size_results[clf_name][frac] = res
            print(f"  {clf_name} ({frac*100:.0f}% data, n={n}): "
                  f"EER={res['eer']:.4f}  AUC={res['auc']:.4f}")

    plot_trainsize_effect(size_results, os.path.join(OUTPUT_DIR, 'trainsize_effect.png'))

    # ---- 儲存結果報告 ----
    print("\n" + "=" * 70)
    print("最終結果摘要")
    print("=" * 70)

    # 結果表格
    header = f"{'分類器':<25s} {'EER':>8s} {'AUC':>8s} {'minDCF':>8s} {'Acc':>8s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Time':>8s}"
    print(f"\n--- Val set (1:1 正負比) ---")
    print(header)
    print("-" * len(header))

    report_lines = []
    report_lines.append("Pair-wise Speaker Verification Baseline 比較實驗結果")
    report_lines.append("=" * 70)
    report_lines.append(f"資料集: tse_aishell")
    report_lines.append(f"Train: {len(train_samples)} 樣本, {len(train_spk)} 說話者")
    report_lines.append(f"Val:   {len(val_samples)} 樣本, {len(val_spk)} 說話者")
    report_lines.append(f"Train pairs: {len(y_train)} (正={y_train.sum()}, 負={len(y_train)-y_train.sum()})")
    report_lines.append(f"Val pairs (1:1): {len(y_val)} (正={y_val.sum()}, 負={len(y_val)-y_val.sum()})")
    report_lines.append(f"Val pairs (1:3): {len(y_val3)} (正={y_val3.sum()}, 負={len(y_val3)-y_val3.sum()})")
    report_lines.append(f"Pair 特徵: |diff| + element-wise product + cosine (385維)")
    report_lines.append("")
    report_lines.append("--- Val set (1:1) ---")
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for res in all_results_1to1:
        line = (f"{res['name']:<25s} {res['eer']:>8.4f} {res['auc']:>8.4f} "
                f"{res['min_dcf']:>8.4f} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
                f"{res['recall']:>8.4f} {res['f1']:>8.4f} {res.get('train_time', 0):>7.1f}s")
        print(line)
        report_lines.append(line)

    print(f"\n--- Val set (1:3 正負比) ---")
    print(header)
    print("-" * len(header))
    report_lines.append("")
    report_lines.append("--- Val set (1:3) ---")
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for res in all_results_1to3:
        line = (f"{res['name']:<25s} {res['eer']:>8.4f} {res['auc']:>8.4f} "
                f"{res['min_dcf']:>8.4f} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
                f"{res['recall']:>8.4f} {res['f1']:>8.4f}")
        print(line)
        report_lines.append(line)

    # 儲存文字報告
    report_path = os.path.join(OUTPUT_DIR, 'results_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n報告已儲存: {report_path}")

    # 儲存數值結果 (JSON)
    json_results = {
        'dataset_info': {
            'train_samples': len(train_samples),
            'train_speakers': len(train_spk),
            'val_samples': len(val_samples),
            'val_speakers': len(val_spk),
            'train_pairs': int(len(y_train)),
            'val_pairs_1to1': int(len(y_val)),
            'val_pairs_1to3': int(len(y_val3)),
        },
        'results_1to1': [],
        'results_1to3': [],
        'feature_comparison': {},
        'trainsize_effect': {},
    }

    for res in all_results_1to1:
        json_results['results_1to1'].append({
            'name': res['name'],
            'eer': float(res['eer']),
            'auc': float(res['auc']),
            'min_dcf': float(res['min_dcf']),
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1': float(res['f1']),
            'threshold': float(res['threshold']),
            'train_time': float(res.get('train_time', 0)),
        })

    for res in all_results_1to3:
        json_results['results_1to3'].append({
            'name': res['name'],
            'eer': float(res['eer']),
            'auc': float(res['auc']),
            'min_dcf': float(res['min_dcf']),
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1': float(res['f1']),
        })

    for clf_name in feat_results:
        json_results['feature_comparison'][clf_name] = {}
        for fm in feat_results[clf_name]:
            r = feat_results[clf_name][fm]
            json_results['feature_comparison'][clf_name][fm] = {
                'eer': float(r['eer']), 'auc': float(r['auc'])
            }

    for clf_name in size_results:
        json_results['trainsize_effect'][clf_name] = {}
        for frac in size_results[clf_name]:
            r = size_results[clf_name][frac]
            json_results['trainsize_effect'][clf_name][str(frac)] = {
                'eer': float(r['eer']), 'auc': float(r['auc'])
            }

    json_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"JSON 結果已儲存: {json_path}")

    print("\n" + "=" * 70)
    print("所有實驗完成！")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
