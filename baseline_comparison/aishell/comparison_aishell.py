#!/usr/bin/env python3
"""
Baseline Classifier Comparison on USEF-TSE tse_aishell Dataset
=============================================================
Compares speaker verification classifiers using WeSpeaker embeddings (192-dim).

Task: Binary classification — given an enrollment embedding, determine if it
belongs to a target speaker. Extremely imbalanced (~2-4 positive vs ~488 negative).

Strategy: Use Leave-One-Out CV (robust for few-shot). Evaluate via scoring
functions (not hard predictions) for ROC/EER. Use class_weight='balanced'
everywhere applicable.

Classifiers: Cosine Similarity, MLP, SVM-RBF, SVM-Linear,
             Logistic Regression, KNN

Extra: 10 random target speakers averaged for statistical significance.
       ONNX vs metadata embedding consistency check.
"""

import os
import sys
import json
import glob
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.base import clone
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────
N_FOLDS = 5
N_TARGET_SPEAKERS = 10
RANDOM_SEED = 42
EMBEDDING_KEY = "speaker_embedding"


def find_paths():
    """Auto-detect paths based on common locations."""
    script_dir = Path(__file__).resolve().parent

    candidates_val = [
        Path("C:/Users/featuer/Desktop/datasets_copy/tse_aishell/val"),
        Path("/sessions/peaceful-sweet-babbage/mnt/datasets_copy/tse_aishell/val"),
    ]
    val_dir = next((c for c in candidates_val if c.exists()), None)

    candidates_onnx = [
        Path("C:/Users/featuer/Desktop/pvad-se-pipeline/models/ecapa_tdnn/wespeaker_resnet34.onnx"),
        Path("/sessions/peaceful-sweet-babbage/mnt/pvad-se-pipeline/models/ecapa_tdnn/wespeaker_resnet34.onnx"),
    ]
    onnx_path = next((c for c in candidates_onnx if c.exists()), None)

    return val_dir, script_dir, onnx_path


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = 0.5
    return eer


def load_val_data(val_dir):
    """Load all val samples."""
    samples = []
    sample_dirs = sorted(glob.glob(str(Path(val_dir) / "sample_*")))

    for sd in sample_dirs:
        meta_path = os.path.join(sd, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        emb = np.array(meta[EMBEDDING_KEY], dtype=np.float32)
        samples.append({
            'speaker_id': meta['speaker_id'],
            'embedding': emb,
            'path': sd,
            'interference_speaker': meta.get('interference_speaker', ''),
            'snr_db': meta.get('snr_db', 0.0),
        })

    speakers = set(s['speaker_id'] for s in samples)
    print(f"Loaded {len(samples)} val samples, {len(speakers)} unique speakers")
    return samples


def prepare_binary_data(samples, target_speaker_id):
    """Prepare binary classification data."""
    X = np.array([s['embedding'] for s in samples])
    y = np.array([1 if s['speaker_id'] == target_speaker_id else 0
                  for s in samples])
    return X, y


class CosineClassifier:
    """Threshold-based cosine similarity classifier."""

    def __init__(self):
        self.centroid = None
        self.threshold = 0.5

    def fit(self, X, y):
        pos_mask = y == 1
        if pos_mask.sum() == 0:
            self.centroid = np.zeros(X.shape[1])
        else:
            self.centroid = X[pos_mask].mean(axis=0)
            self.centroid /= (np.linalg.norm(self.centroid) + 1e-8)

        # Optimize threshold
        scores = self.decision_function(X)
        best_f1, best_t = 0, 0.5
        for t in np.linspace(scores.min(), scores.max(), 200):
            preds = (scores >= t).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue
            f = f1_score(y, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        self.threshold = best_t

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(int)

    def decision_function(self, X):
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        c_norm = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)
        return X_norm @ c_norm


def get_classifiers(n_positive):
    """Return classifiers, adapting to available data."""
    k = max(1, min(3, n_positive - 1))  # KNN k < n_positive in train

    return {
        'Cosine Similarity': CosineClassifier(),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500,
                early_stopping=False,  # disable to avoid needing val split
                random_state=RANDOM_SEED,
                learning_rate='adaptive',
                alpha=0.01,  # strong regularization for few-shot
            ))
        ]),
        'SVM-RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True,
                       class_weight='balanced', random_state=RANDOM_SEED,
                       gamma='scale'))
        ]),
        'SVM-Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', probability=True,
                       class_weight='balanced', random_state=RANDOM_SEED))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced', max_iter=1000,
                random_state=RANDOM_SEED, C=0.1
            ))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=k, weights='distance'))
        ]),
    }


def get_scores(clf, X):
    """Get continuous scores for ROC/EER."""
    if isinstance(clf, CosineClassifier):
        return clf.decision_function(X)
    # For pipelines, check the final estimator
    final = clf[-1] if hasattr(clf, '__getitem__') else clf
    if hasattr(final, 'decision_function'):
        return clf.decision_function(X)
    elif hasattr(final, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    else:
        return clf.predict(X).astype(float)


def run_single_speaker_experiment(samples, target_speaker_id):
    """
    Run LOO CV for all classifiers on one target speaker.
    For efficiency with 492 samples, we use stratified 5-fold instead of
    pure LOO (which would be 492 iterations × 6 classifiers).
    """
    X, y = prepare_binary_data(samples, target_speaker_id)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())

    if n_pos < 2:
        return None

    # Use stratified K-fold with K = min(5, n_pos)
    effective_folds = min(N_FOLDS, n_pos)
    if effective_folds < 2:
        effective_folds = 2

    classifiers = get_classifiers(n_pos)
    results = {}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True,
                          random_state=RANDOM_SEED)

    for clf_name, clf_template in classifiers.items():
        all_y_true = []
        all_y_pred = []
        all_y_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if y_train.sum() == 0:
                continue

            # Fresh classifier
            if isinstance(clf_template, CosineClassifier):
                clf = CosineClassifier()
            else:
                clf = clone(clf_template)

            try:
                clf.fit(X_train, y_train)
            except Exception as e:
                print(f"    {clf_name} fold {fold_idx} fit error: {e}")
                continue

            y_pred = clf.predict(X_test)
            scores = get_scores(clf, X_test)

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_scores.extend(scores.tolist())

        if not all_y_true or sum(all_y_true) == 0:
            continue

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_scores = np.array(all_y_scores)

        eer = compute_eer(all_y_true, all_y_scores)

        results[clf_name] = {
            'accuracy': accuracy_score(all_y_true, all_y_pred),
            'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
            'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
            'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
            'eer': eer,
            'y_true': all_y_true,
            'y_scores': all_y_scores,
            'n_positive': n_pos,
            'n_negative': n_neg,
        }

    return results if results else None


def plot_roc_curves(results, title, output_path):
    """Plot ROC curves for all classifiers."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (clf_name, metrics), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_scores'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{clf_name} (AUC={roc_auc:.3f}, EER={metrics["eer"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_multi_speaker_summary(all_speaker_results, output_path):
    """Bar chart comparing classifiers across speakers."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'eer']
    clf_names = sorted(
        set(c for r in all_speaker_results.values() for c in r.keys()),
        key=lambda x: list(get_classifiers(5).keys()).index(x)
        if x in get_classifiers(5) else 99
    )

    agg = {clf: {m: [] for m in metrics_to_plot} for clf in clf_names}
    for results in all_speaker_results.values():
        for clf_name in clf_names:
            if clf_name in results:
                for m in metrics_to_plot:
                    agg[clf_name][m].append(results[clf_name][m])

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    x = np.arange(len(clf_names))
    width = 0.6
    colors = plt.cm.Set2(np.linspace(0, 1, len(clf_names)))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        means = [np.mean(agg[clf][metric]) if agg[clf][metric] else 0
                 for clf in clf_names]
        stds = [np.std(agg[clf][metric]) if agg[clf][metric] else 0
                for clf in clf_names]

        bars = ax.bar(x, means, width, yerr=stds, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric.upper(), fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in clf_names],
                           fontsize=7, rotation=0)
        ax.set_ylim([0, 1.15])
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Classifier Comparison on tse_aishell (Avg over 10 speakers)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary plot: {output_path}")


def plot_averaged_roc(all_speaker_results, output_path):
    """Mean ROC curves averaged across speakers."""
    clf_names = sorted(
        set(c for r in all_speaker_results.values() for c in r.keys()),
        key=lambda x: list(get_classifiers(5).keys()).index(x)
        if x in get_classifiers(5) else 99
    )
    colors = plt.cm.Set2(np.linspace(0, 1, len(clf_names)))

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    mean_fpr = np.linspace(0, 1, 200)

    for clf_name, color in zip(clf_names, colors):
        tprs = []
        aucs = []
        eers = []
        for results in all_speaker_results.values():
            if clf_name not in results:
                continue
            y_true = results[clf_name]['y_true']
            y_scores = results[clf_name]['y_scores']
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            interp_tpr = interp1d(fpr, tpr, bounds_error=False,
                                  fill_value=(0, 1))(mean_fpr)
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
            eers.append(results[clf_name]['eer'])

        if not tprs:
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_eer = np.mean(eers)
        std_eer = np.std(eers)

        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f'{clf_name}\n  AUC={mean_auc:.3f}±{std_auc:.3f}, '
                      f'EER={mean_eer:.3f}±{std_eer:.3f}')

        std_tpr = np.std(tprs, axis=0)
        ax.fill_between(mean_fpr,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=0.1)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Mean ROC Curves (Averaged over 10 Target Speakers)', fontsize=14)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved averaged ROC plot: {output_path}")


def _compute_fbank(audio, sr=16000, n_mels=80, n_fft=512,
                   win_length=400, hop_length=160):
    """Compute Kaldi-style Fbank features matching WeSpeaker/ECAPA-TDNN."""
    import librosa
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window='hamming', center=False)
    power_spec = np.abs(stft) ** 2
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                  fmin=20, fmax=sr // 2).astype(np.float32)
    mel = mel_fb @ power_spec
    log_mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
    log_mel = log_mel - log_mel.mean(axis=1, keepdims=True)  # CMN
    return log_mel  # (n_mels, T)


def run_onnx_consistency_check(samples, onnx_path, n_check=20):
    """
    Compare metadata embeddings vs ONNX-extracted embeddings.
    Uses ecapa_tdnn.onnx (192-dim, input: batch,80,T) by default.
    Falls back to wespeaker_resnet34.onnx (256-dim, input: batch,T,80) if needed.
    """
    print("\n" + "=" * 60)
    print("ONNX Consistency Check")
    print("=" * 60)

    try:
        import onnxruntime as ort
        import soundfile as sf
        import librosa
    except ImportError as e:
        print(f"  Skipping: {e}")
        return None

    # Find the right model - prefer ecapa_tdnn (192-dim matches metadata)
    ecapa_path = None
    if onnx_path:
        ecapa_candidate = Path(onnx_path).parent / "ecapa_tdnn.onnx"
        if ecapa_candidate.exists():
            ecapa_path = ecapa_candidate

    model_path = ecapa_path or onnx_path
    if model_path is None or not Path(model_path).exists():
        print(f"  Skipping: ONNX model not found")
        return None

    session = ort.InferenceSession(str(model_path))
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # e.g. ['batch', 80, 'time'] or ['B', 'T', 80]

    # Determine input format
    # ecapa_tdnn: (batch, 80, time) - mel_features
    # wespeaker_resnet34: (batch, time, 80) - feats
    is_ecapa_format = (len(input_shape) == 3 and
                       (input_shape[1] == 80 or input_name == 'mel_features'))

    out_dim = output_info.shape[-1] if isinstance(output_info.shape[-1], int) else 192
    print(f"  Model: {Path(model_path).name}")
    print(f"  Input: {input_name} {input_shape}")
    print(f"  Output dim: {out_dim}")
    print(f"  Format: {'(B,80,T)' if is_ecapa_format else '(B,T,80)'}")

    # Check dimension match with metadata
    meta_dim = len(samples[0]['embedding'])
    if out_dim != meta_dim:
        print(f"  WARNING: Model output dim ({out_dim}) != metadata dim ({meta_dim})")
        print(f"  Results may not match.")

    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(samples), min(n_check, len(samples)),
                               replace=False)

    cosine_sims = []
    l2_dists = []

    for idx in indices:
        sample = samples[idx]
        enrollment_path = os.path.join(sample['path'], 'enrollment.wav')
        if not os.path.exists(enrollment_path):
            continue
        try:
            audio, sr = sf.read(enrollment_path)
            if sr != 16000:
                audio = librosa.resample(audio.astype(np.float32),
                                         orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)

            # Compute Fbank features
            fbank = _compute_fbank(audio)  # (80, T)

            if is_ecapa_format:
                # ecapa_tdnn expects (batch, 80, T)
                fbank_input = fbank[np.newaxis, :, :]  # (1, 80, T)
            else:
                # wespeaker expects (batch, T, 80)
                fbank_input = fbank.T[np.newaxis, :, :]  # (1, T, 80)

            result = session.run(None, {input_name: fbank_input})
            onnx_emb = result[0].flatten()[:meta_dim]  # trim to meta dim
            onnx_emb /= (np.linalg.norm(onnx_emb) + 1e-8)

            meta_emb = sample['embedding'].copy()
            meta_emb /= (np.linalg.norm(meta_emb) + 1e-8)

            cos = float(np.dot(onnx_emb, meta_emb))
            l2 = float(np.linalg.norm(onnx_emb - meta_emb))
            cosine_sims.append(cos)
            l2_dists.append(l2)

            print(f"  [{sample['speaker_id']}] cos={cos:.6f}, L2={l2:.6f}")
        except Exception as e:
            print(f"  Error on sample {idx}: {e}")

    if cosine_sims:
        result = {
            'cosine_sim_mean': float(np.mean(cosine_sims)),
            'cosine_sim_std': float(np.std(cosine_sims)),
            'l2_dist_mean': float(np.mean(l2_dists)),
            'l2_dist_std': float(np.std(l2_dists)),
            'n_checked': len(cosine_sims),
            'consistent': np.mean(cosine_sims) > 0.99,
            'model_used': str(Path(model_path).name),
        }
        print(f"\n  Cosine: {result['cosine_sim_mean']:.6f} ± {result['cosine_sim_std']:.6f}")
        print(f"  L2:     {result['l2_dist_mean']:.6f} ± {result['l2_dist_std']:.6f}")
        print(f"  Consistent: {'YES' if result['consistent'] else 'NO'}")
        return result
    return None


def generate_report(all_speaker_results, onnx_check, output_dir):
    """Generate text report."""
    clf_names = sorted(
        set(c for r in all_speaker_results.values() for c in r.keys()),
        key=lambda x: list(get_classifiers(5).keys()).index(x)
        if x in get_classifiers(5) else 99
    )
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'eer']

    agg = {clf: {m: [] for m in metrics_names} for clf in clf_names}
    for results in all_speaker_results.values():
        for clf_name in clf_names:
            if clf_name in results:
                for m in metrics_names:
                    agg[clf_name][m].append(results[clf_name][m])

    lines = []
    lines.append("=" * 72)
    lines.append("Baseline Classifier Comparison Report")
    lines.append("Dataset: USEF-TSE tse_aishell (val set, 492 samples)")
    lines.append(f"Target speakers tested: {len(all_speaker_results)}")
    lines.append(f"CV: Stratified K-Fold (K = min(5, n_positive))")
    lines.append("=" * 72)
    lines.append("")

    for spk_idx, (spk_id, results) in enumerate(all_speaker_results.items()):
        n_pos = next(iter(results.values()))['n_positive']
        n_neg = next(iter(results.values()))['n_negative']
        lines.append(f"--- Speaker #{spk_idx+1}: {spk_id} "
                     f"(pos={n_pos}, neg={n_neg}) ---")
        lines.append(f"  {'Classifier':<22} {'Acc':>7} {'Prec':>7} "
                     f"{'Rec':>7} {'F1':>7} {'EER':>7}")
        for clf_name in clf_names:
            if clf_name in results:
                m = results[clf_name]
                lines.append(f"  {clf_name:<22} {m['accuracy']:>7.4f} "
                           f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
                           f"{m['f1']:>7.4f} {m['eer']:>7.4f}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("AGGREGATED RESULTS (mean ± std across speakers)")
    lines.append("=" * 72)
    header = f"  {'Classifier':<22}"
    for m in metrics_names:
        header += f" {m.upper():>14}"
    lines.append(header)

    for clf_name in clf_names:
        row = f"  {clf_name:<22}"
        for m in metrics_names:
            vals = agg[clf_name][m]
            if vals:
                row += f" {np.mean(vals):>.4f}±{np.std(vals):.3f}"
            else:
                row += f" {'N/A':>14}"
        lines.append(row)

    if onnx_check:
        lines.append("")
        lines.append("=" * 72)
        lines.append("ONNX CONSISTENCY CHECK")
        lines.append("=" * 72)
        lines.append(f"  Samples: {onnx_check['n_checked']}")
        lines.append(f"  Cosine:  {onnx_check['cosine_sim_mean']:.6f} "
                     f"± {onnx_check['cosine_sim_std']:.6f}")
        lines.append(f"  L2:      {onnx_check['l2_dist_mean']:.6f} "
                     f"± {onnx_check['l2_dist_std']:.6f}")
        lines.append(f"  Match:   {'YES' if onnx_check['consistent'] else 'NO'}")

    report = "\n".join(lines)
    path = os.path.join(output_dir, "report_aishell.txt")
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Saved report: {path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--onnx-model', type=str, default=None)
    parser.add_argument('--n-speakers', type=int, default=N_TARGET_SPEAKERS)
    parser.add_argument('--skip-onnx-check', action='store_true')
    args = parser.parse_args()

    auto_val, auto_output, auto_onnx = find_paths()
    val_dir = args.val_dir or auto_val
    output_dir = args.output_dir or auto_output
    onnx_path = args.onnx_model or auto_onnx

    if val_dir is None or not Path(val_dir).exists():
        print("ERROR: Val directory not found. Use --val-dir")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Baseline Classifier Comparison - tse_aishell")
    print("=" * 60)
    print(f"Val dir:    {val_dir}")
    print(f"Output dir: {output_dir}")
    print(f"ONNX model: {onnx_path}")
    print()

    samples = load_val_data(val_dir)

    # Find speakers with >= 2 samples
    speaker_counts = defaultdict(int)
    for s in samples:
        speaker_counts[s['speaker_id']] += 1
    eligible = [spk for spk, cnt in speaker_counts.items() if cnt >= 2]
    print(f"Speakers with >= 2 samples: {len(eligible)}")

    # Prefer speakers with more samples for more meaningful results
    eligible_sorted = sorted(eligible, key=lambda s: speaker_counts[s],
                             reverse=True)

    np.random.seed(RANDOM_SEED)
    n_target = min(args.n_speakers, len(eligible_sorted))

    # Take top-half by count, then random sample from those
    top_half = eligible_sorted[:max(len(eligible_sorted)//2, n_target)]
    target_speakers = list(np.random.choice(top_half, n_target, replace=False))

    print(f"Selected {n_target} target speakers:")
    for spk in target_speakers:
        print(f"  {spk}: {speaker_counts[spk]} samples")
    print()

    # ─── Run experiments ─────────────────────────────────────────
    all_speaker_results = {}

    for i, spk_id in enumerate(target_speakers):
        print(f"[{i+1}/{n_target}] Speaker: {spk_id} "
              f"({speaker_counts[spk_id]} samples)")

        results = run_single_speaker_experiment(samples, spk_id)

        if results is None:
            print(f"  Skipped")
            continue

        all_speaker_results[spk_id] = results

        print(f"  {'Classifier':<22} {'Acc':>7} {'Prec':>7} "
              f"{'Rec':>7} {'F1':>7} {'EER':>7}")
        for clf_name, m in results.items():
            print(f"  {clf_name:<22} {m['accuracy']:>7.4f} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
                  f"{m['f1']:>7.4f} {m['eer']:>7.4f}")

        roc_path = os.path.join(output_dir, f"roc_{spk_id}.png")
        plot_roc_curves(results, f"ROC - Target: {spk_id}", roc_path)
        print(f"  Saved: {roc_path}")

    if not all_speaker_results:
        print("ERROR: No valid results.")
        sys.exit(1)

    # ─── Summary ─────────────────────────────────────────────────
    print("\nGenerating summary...")
    plot_multi_speaker_summary(
        all_speaker_results,
        os.path.join(output_dir, "summary_metrics.png")
    )
    plot_averaged_roc(
        all_speaker_results,
        os.path.join(output_dir, "roc_averaged.png")
    )

    # ─── ONNX check ──────────────────────────────────────────────
    onnx_check = None
    if not args.skip_onnx_check:
        onnx_check = run_onnx_consistency_check(samples, onnx_path)

    # ─── Report ──────────────────────────────────────────────────
    report = generate_report(all_speaker_results, onnx_check, output_dir)
    print("\n" + report)

    # Save JSON
    json_out = {}
    for spk_id, results in all_speaker_results.items():
        json_out[spk_id] = {
            clf: {k: v for k, v in m.items() if k not in ('y_true', 'y_scores')}
            for clf, m in results.items()
        }
    json_path = os.path.join(output_dir, "results_aishell.json")
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved: {json_path}")
    print("Done!")


if __name__ == '__main__':
    main()
