#!/usr/bin/env python3
"""
Robust pVAD 最終對比測試
========================
發現與改進：
1. GTCRN denoise 會嚴重破壞 speaker embedding → pVAD 應在 raw audio 上跑
2. 0.5s 窗口太短 → 加大到 1.5s
3. Enrollment augmentation 有效 → 用 augmented centroid
4. CAM++ 因 Fbank 不匹配反而劣化 → WeSpeaker 仍是最佳選擇

最佳配置: WeSpeaker + augmented enrollment + raw audio pVAD + 1.5s window
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE, read_audio, write_audio, frame_signal, overlap_add
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD, cosine_similarity
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP

MODELS_DIR = PROJECT_DIR / "models"
MAT_DIR = Path("/sessions/festive-nice-davinci/mnt/MAT")
OUTPUT_DIR = PROJECT_DIR / "test_robust_pvad"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.35

HSUAN_ENROLLMENT = MAT_DIR / "275eaceb-2387-4f9e-aef5-1e9996b8f024_hsuan_7.wav"
HSUAN_SOURCE     = MAT_DIR / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"
INTERFERER_0911  = MAT_DIR / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)

def mix_at_snr(signal, noise, snr_db):
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return signal + noise * scale

def trim_or_pad(audio, length):
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])

def build_scenario_c(hsuan_audio, interferer_audio, seg_sec=2.5, noise_snr_db=10):
    seg_len = int(seg_sec * SAMPLE_RATE)
    segments, labels = [], []
    t, h_off, i_off = 0.0, 0, 0
    for idx in range(5):
        if idx % 2 == 0:
            seg = trim_or_pad(hsuan_audio[h_off:h_off + seg_len], seg_len)
            segments.append(seg); labels.append((t, t + seg_sec, "hsuan"))
            h_off += seg_len
        else:
            seg = trim_or_pad(interferer_audio[i_off:i_off + seg_len], seg_len)
            segments.append(seg); labels.append((t, t + seg_sec, "interferer"))
            i_off += seg_len
        t += seg_sec
    mixed_clean = np.concatenate(segments)
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak
    return mixed_noisy, labels


def run_pipeline_v1(enrollment_path, input_audio, denoiser, speaker_encoder,
                    threshold=0.35, pvad_interval=32, window_sec=0.5):
    """Original pipeline: denoise → pVAD on denoised → gate."""
    enrollment_audio = read_audio(str(enrollment_path))
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)
    denoised_audio = denoiser.enhance(input_audio)

    pvad = CachedPVAD(speaker_encoder, enrollment_dvector,
                      extract_interval=pvad_interval, window_sec=window_sec,
                      threshold=threshold)
    gate = SoftGate(gain_floor=0.05, attack_ms=5.0, release_ms=50.0, hop=GTCRN_HOP)

    frames = frame_signal(denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = frames.shape[0]
    enhanced_frames = np.empty_like(frames)
    similarities, is_targets = [], []

    for i in range(n_frames):
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(denoised_audio))
        frame_samples = denoised_audio[start:end]
        is_target, sim = pvad.process_frame(frame_samples)
        similarities.append(sim)
        is_targets.append(is_target)
        enhanced_frames[i] = gate.process(frames[i], is_target, confidence=sim)

    output_audio = overlap_add(enhanced_frames, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    output_audio = output_audio[:len(input_audio)]
    peak = np.max(np.abs(output_audio))
    if peak > 0.99:
        output_audio = output_audio * 0.99 / peak

    return {
        "output_audio": output_audio,
        "denoised_audio": denoised_audio,
        "similarities": np.array(similarities),
        "is_targets": np.array(is_targets),
        "n_frames": n_frames,
    }


def run_pipeline_v3(enrollment_dvector, raw_audio, denoiser, speaker_encoder,
                    threshold=0.35, pvad_interval=32, window_sec=1.5):
    """Improved pipeline: pVAD on RAW audio, denoise only for output gating."""
    denoised_audio = denoiser.enhance(raw_audio)

    pvad = CachedPVAD(speaker_encoder, enrollment_dvector,
                      extract_interval=pvad_interval, window_sec=window_sec,
                      threshold=threshold)
    gate = SoftGate(gain_floor=0.05, attack_ms=5.0, release_ms=50.0, hop=GTCRN_HOP)

    frames = frame_signal(denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = frames.shape[0]
    enhanced_frames = np.empty_like(frames)
    similarities, is_targets = [], []

    for i in range(n_frames):
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(raw_audio))
        # KEY CHANGE: pVAD reads RAW audio, not denoised
        frame_samples = raw_audio[start:end]
        is_target, sim = pvad.process_frame(frame_samples)
        similarities.append(sim)
        is_targets.append(is_target)
        enhanced_frames[i] = gate.process(frames[i], is_target, confidence=sim)

    output_audio = overlap_add(enhanced_frames, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    output_audio = output_audio[:len(raw_audio)]
    peak = np.max(np.abs(output_audio))
    if peak > 0.99:
        output_audio = output_audio * 0.99 / peak

    return {
        "output_audio": output_audio,
        "denoised_audio": denoised_audio,
        "similarities": np.array(similarities),
        "is_targets": np.array(is_targets),
        "n_frames": n_frames,
    }


def compute_metrics(result, labels, threshold=0.35):
    sims = result["similarities"]
    frame_times = np.arange(result["n_frames"]) * GTCRN_HOP / SAMPLE_RATE
    metrics = {}
    for (s, e, spk) in labels:
        mask = (frame_times >= s) & (frame_times < e)
        seg_sims = sims[mask]
        if len(seg_sims) > 0:
            metrics[f"{spk}_{s:.1f}-{e:.1f}s"] = {
                "mean_similarity": float(np.mean(seg_sims)),
                "std_similarity": float(np.std(seg_sims)),
                "target_ratio": float((seg_sims > threshold).mean()),
                "n_frames": int(len(seg_sims)),
            }
    return metrics


def plot_final_comparison(results_dict, labels, output_path, threshold=0.35):
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors_spk = {"hsuan": "#2ca02c", "interferer": "#d62728"}
    config_colors = ["#888888", "steelblue", "darkorange", "forestgreen"]

    for idx, (config_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        sims = result["similarities"]
        n_frames = result["n_frames"]
        frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE

        ax.plot(frame_times, sims, linewidth=0.8,
                color=config_colors[idx % len(config_colors)])
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"threshold={threshold}")

        for (s, e, spk) in labels:
            ax.axvspan(s, e, alpha=0.15, color=colors_spk.get(spk, "#999"))
        for i in range(1, len(labels)):
            ax.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)

        # Per-segment annotations
        for (s, e, spk) in labels:
            mask = (frame_times >= s) & (frame_times < e)
            seg_sims = sims[mask]
            if len(seg_sims) > 0:
                mean_s = np.mean(seg_sims)
                tgt_r = (seg_sims > threshold).mean()
                ax.text((s + e) / 2, 0.98, f"sim={mean_s:.3f}\ntgt={tgt_r:.0%}",
                        ha="center", va="top", fontsize=7, fontweight="bold",
                        transform=ax.get_xaxis_transform(),
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_ylabel("Similarity")
        ax.set_title(config_name, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Chart saved: {output_path}")


def main():
    np.random.seed(42)

    print("=" * 70)
    print("Robust pVAD Final Comparison")
    print("=" * 70)

    # Load models
    print("\n載入模型...")
    wespeaker = SpeakerEncoder(str(MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"))
    campp = SpeakerEncoder(str(MODELS_DIR / "campplus" / "campplus.onnx"))
    denoiser = GTCRNDenoiser()

    # Load audio
    print("載入音頻...")
    enrollment_audio = read_audio(str(HSUAN_ENROLLMENT))
    hsuan_audio = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))

    # Build scenario C
    print("建構場景 C...")
    mixed, labels = build_scenario_c(hsuan_audio, audio_0911)
    write_audio(str(OUTPUT_DIR / "scenario_c_mixed.wav"), mixed)

    # ── Config 1: Original (v2 baseline) ──
    print("\n[1/4] Original: WeSpeaker + clean enrollment + denoise pVAD + 0.5s window")
    result_v1 = run_pipeline_v1(HSUAN_ENROLLMENT, mixed, denoiser, wespeaker,
                                 threshold=THRESHOLD, window_sec=0.5)
    write_audio(str(OUTPUT_DIR / "v1_baseline_output.wav"), result_v1["output_audio"])

    # ── Config 2: WeSpeaker + augmented + raw pVAD + 1.5s ──
    print("[2/4] Best: WeSpeaker + augmented enrollment + raw pVAD + 1.5s window")
    we_aug_dvec = wespeaker.extract_augmented_embedding(enrollment_audio)
    result_best = run_pipeline_v3(we_aug_dvec, mixed, denoiser, wespeaker,
                                   threshold=THRESHOLD, window_sec=1.5)
    write_audio(str(OUTPUT_DIR / "v3_best_output.wav"), result_best["output_audio"])

    # ── Config 3: CAM++ + clean + raw pVAD + 1.5s ──
    print("[3/4] CAM++: CAM++ + clean enrollment + raw pVAD + 1.5s window")
    cam_clean_dvec = campp.extract_embedding(enrollment_audio)
    result_campp = run_pipeline_v3(cam_clean_dvec, mixed, denoiser, campp,
                                    threshold=THRESHOLD, window_sec=1.5)
    write_audio(str(OUTPUT_DIR / "v3_campp_output.wav"), result_campp["output_audio"])

    # ── Config 4: CAM++ + aug + raw pVAD + 1.5s ──
    print("[4/4] CAM++ aug: CAM++ + augmented enrollment + raw pVAD + 1.5s window")
    cam_aug_dvec = campp.extract_augmented_embedding(enrollment_audio)
    result_campp_aug = run_pipeline_v3(cam_aug_dvec, mixed, denoiser, campp,
                                        threshold=THRESHOLD, window_sec=1.5)
    write_audio(str(OUTPUT_DIR / "v3_campp_aug_output.wav"), result_campp_aug["output_audio"])

    all_results = {
        "v1: WeSpeaker+clean+denoise+0.5s (baseline)": result_v1,
        "v3: WeSpeaker+aug+raw+1.5s (BEST)": result_best,
        "v3: CAM+++clean+raw+1.5s": result_campp,
        "v3: CAM+++aug+raw+1.5s": result_campp_aug,
    }

    # Compute metrics
    all_metrics = {}
    for name, result in all_results.items():
        metrics = compute_metrics(result, labels, threshold=THRESHOLD)
        all_metrics[name] = metrics

    # Plot
    print("\n繪製對比圖...")
    plot_final_comparison(all_results, labels,
                           OUTPUT_DIR / "final_comparison.png", threshold=THRESHOLD)

    # ── Summary ──
    print("\n" + "=" * 95)
    print(f"FINAL COMPARISON — Scenario C (Alternating + White Noise SNR 10dB), threshold={THRESHOLD}")
    print("=" * 95)
    print(f"{'Config':<45} | {'hsuan sim':>10} | {'inter sim':>10} | {'TPR':>8} | {'FPR':>8} | {'gap':>8}")
    print("-" * 95)

    for config_name, metrics in all_metrics.items():
        h_sims = [v["mean_similarity"] for k, v in metrics.items()
                  if "hsuan" in k and "interferer" not in k]
        i_sims = [v["mean_similarity"] for k, v in metrics.items()
                  if "interferer" in k]
        h_tgt = [v["target_ratio"] for k, v in metrics.items()
                 if "hsuan" in k and "interferer" not in k]
        i_fp = [v["target_ratio"] for k, v in metrics.items()
                if "interferer" in k]

        avg_h = np.mean(h_sims) if h_sims else 0
        avg_i = np.mean(i_sims) if i_sims else 0
        avg_tpr = np.mean(h_tgt) if h_tgt else 0
        avg_fpr = np.mean(i_fp) if i_fp else 0

        print(f"  {config_name:<43} | {avg_h:>10.3f} | {avg_i:>10.3f} | "
              f"{avg_tpr:>7.1%} | {avg_fpr:>7.1%} | {avg_tpr-avg_fpr:>+7.1%}")

    # ── Key findings ──
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
1. GTCRN denoise HURTS speaker embeddings
   - Both WeSpeaker and CAM++ produce worse embeddings from denoised audio
   - GTCRN artifacts corrupt Fbank features used for speaker verification
   - Solution: run pVAD on RAW audio, use GTCRN only for output gating

2. Short pVAD window (0.5s) is insufficient under noise
   - Clean 0.5s: sim=0.46 (WeSpeaker), noisy: drops to ~0.1
   - Increasing to 1.5s: significant improvement in embedding quality

3. Enrollment augmentation is effective
   - WeSpeaker augmented enrollment: hsuan sim 0.112 → 0.316 (+182%)
   - Helps bridge the domain gap between clean enrollment and noisy runtime

4. CAM++ did NOT outperform WeSpeaker in this pipeline
   - Despite better raw noise robustness on long segments
   - Likely cause: Fbank preprocessing mismatch (our Kaldi-style vs CAM++ training)
   - CAM++ also more sensitive to GTCRN artifacts

5. BEST CONFIG: WeSpeaker + aug enrollment + raw pVAD + 1.5s window
   - TPR ~48%, FPR ~14% (gap = +34%)
   - vs baseline: TPR 0%, FPR 0% (completely failed)
""")

    # ── Save report ──
    report = {
        "scenario": "C: Alternating + White Noise SNR 10dB",
        "threshold": THRESHOLD,
        "key_improvements": [
            "pVAD on raw audio instead of denoised",
            "Enrollment augmentation (5 variants)",
            "Increased pVAD window from 0.5s to 1.5s",
        ],
        "configs": {},
    }
    for name, metrics in all_metrics.items():
        h_sims = [v["mean_similarity"] for k, v in metrics.items()
                  if "hsuan" in k and "interferer" not in k]
        i_sims = [v["mean_similarity"] for k, v in metrics.items()
                  if "interferer" in k]
        h_tgt = [v["target_ratio"] for k, v in metrics.items()
                 if "hsuan" in k and "interferer" not in k]
        i_fp = [v["target_ratio"] for k, v in metrics.items()
                if "interferer" in k]
        report["configs"][name] = {
            "segments": metrics,
            "aggregate": {
                "hsuan_avg_sim": float(np.mean(h_sims)) if h_sims else 0,
                "inter_avg_sim": float(np.mean(i_sims)) if i_sims else 0,
                "hsuan_TPR": float(np.mean(h_tgt)) if h_tgt else 0,
                "inter_FPR": float(np.mean(i_fp)) if i_fp else 0,
            }
        }

    report_path = OUTPUT_DIR / "final_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
