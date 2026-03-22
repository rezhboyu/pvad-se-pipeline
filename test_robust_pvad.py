#!/usr/bin/env python3
"""
Robust pVAD 測試 — Enrollment Augmentation + CAM++ 對比
=======================================================
場景 C: 交替說話 (hsuan <-> 0911) + 白噪 SNR 10dB

對比三種配置：
  1. baseline: WeSpeaker + clean enrollment (v2 結果重現)
  2. aug_enroll: WeSpeaker + augmented enrollment
  3. campp: CAM++ + clean enrollment
  4. campp_aug: CAM++ + augmented enrollment
"""

import sys
import os
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

# Audio files
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
    """場景 C: 交替說話 + 白噪 SNR 10dB"""
    seg_len = int(seg_sec * SAMPLE_RATE)
    segments, labels = [], []
    t, h_off, i_off = 0.0, 0, 0
    for idx in range(5):
        if idx % 2 == 0:
            seg = trim_or_pad(hsuan_audio[h_off:h_off + seg_len], seg_len)
            segments.append(seg)
            labels.append((t, t + seg_sec, "hsuan"))
            h_off += seg_len
        else:
            seg = trim_or_pad(interferer_audio[i_off:i_off + seg_len], seg_len)
            segments.append(seg)
            labels.append((t, t + seg_sec, "interferer"))
            i_off += seg_len
        t += seg_sec
    mixed_clean = np.concatenate(segments)
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak
    return mixed_noisy, labels


def run_pipeline(enrollment_dvector, input_audio, denoiser, speaker_encoder,
                 threshold=0.35, pvad_interval=32):
    """Run full pipeline: denoise -> pVAD -> gate."""
    denoised_audio = denoiser.enhance(input_audio)

    pvad = CachedPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        threshold=threshold,
    )
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


def plot_comparison(results_dict, labels, output_path, threshold=0.35):
    """Plot similarity timelines for all configurations."""
    n_configs = len(results_dict)
    fig, axes = plt.subplots(n_configs + 1, 1, figsize=(16, 4 * (n_configs + 1)), sharex=True)

    colors_spk = {"hsuan": "#2ca02c", "interferer": "#d62728"}
    config_colors = ["steelblue", "darkorange", "forestgreen", "purple"]

    for idx, (config_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        sims = result["similarities"]
        n_frames = result["n_frames"]
        frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE

        ax.plot(frame_times, sims, linewidth=0.8, color=config_colors[idx % len(config_colors)])
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7)

        for (s, e, spk) in labels:
            ax.axvspan(s, e, alpha=0.15, color=colors_spk.get(spk, "#999"))
        for i in range(1, len(labels)):
            ax.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)

        # Compute per-segment means for annotation
        for (s, e, spk) in labels:
            mask = (frame_times >= s) & (frame_times < e)
            seg_sims = sims[mask]
            if len(seg_sims) > 0:
                mean_sim = np.mean(seg_sims)
                ax.text((s + e) / 2, 0.95, f"{mean_sim:.3f}",
                        ha="center", va="top", fontsize=8, fontweight="bold",
                        transform=ax.get_xaxis_transform())

        ax.set_ylabel("Similarity")
        ax.set_title(f"{config_name}", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Bottom plot: overlay all on same axes
    ax = axes[-1]
    for idx, (config_name, result) in enumerate(results_dict.items()):
        sims = result["similarities"]
        n_frames = result["n_frames"]
        frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE
        ax.plot(frame_times, sims, linewidth=0.8, color=config_colors[idx % len(config_colors)],
                label=config_name, alpha=0.8)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label=f"threshold={threshold}")
    for (s, e, spk) in labels:
        ax.axvspan(s, e, alpha=0.12, color=colors_spk.get(spk, "#999"))
    ax.set_ylabel("Similarity")
    ax.set_xlabel("Time (s)")
    ax.set_title("All Configs Overlay", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Comparison chart saved: {output_path}")


def main():
    np.random.seed(42)

    print("=" * 70)
    print("Robust pVAD 測試 — Scenario C 對比")
    print("=" * 70)

    # ── Load models ──
    print("\n載入模型...")
    wespeaker_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    campp_path = MODELS_DIR / "campplus" / "campplus.onnx"

    wespeaker = SpeakerEncoder(str(wespeaker_path))
    campp = SpeakerEncoder(str(campp_path))
    denoiser = GTCRNDenoiser()
    print(f"  WeSpeaker: embed_dim={wespeaker.embed_dim}")
    print(f"  CAM++:     embed_dim={campp.embed_dim}")
    print("  GTCRN:     loaded")

    # ── Load audio ──
    print("\n載入音頻...")
    enrollment_audio = read_audio(str(HSUAN_ENROLLMENT))
    hsuan_audio = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))
    print(f"  enrollment: {len(enrollment_audio)/SAMPLE_RATE:.1f}s")
    print(f"  hsuan:      {len(hsuan_audio)/SAMPLE_RATE:.1f}s")
    print(f"  0911:       {len(audio_0911)/SAMPLE_RATE:.1f}s")

    # ── Build scenario C ──
    print("\n建構場景 C (交替說話 + 白噪 SNR 10dB)...")
    mixed_audio, labels = build_scenario_c(hsuan_audio, audio_0911)
    write_audio(str(OUTPUT_DIR / "scenario_c_mixed.wav"), mixed_audio)
    print(f"  mixed: {len(mixed_audio)/SAMPLE_RATE:.1f}s")
    print(f"  labels: {labels}")

    # ── Extract enrollment embeddings ──
    print("\n提取 enrollment embeddings...")

    # WeSpeaker clean enrollment
    we_enroll_clean = wespeaker.extract_embedding(enrollment_audio)
    print(f"  WeSpeaker clean: norm={np.linalg.norm(we_enroll_clean):.4f}")

    # WeSpeaker augmented enrollment
    we_enroll_aug = wespeaker.extract_augmented_embedding(enrollment_audio)
    print(f"  WeSpeaker augmented: norm={np.linalg.norm(we_enroll_aug):.4f}")
    print(f"  WeSpeaker clean vs aug cosine: {cosine_similarity(we_enroll_clean, we_enroll_aug):.4f}")

    # CAM++ clean enrollment
    cam_enroll_clean = campp.extract_embedding(enrollment_audio)
    print(f"  CAM++ clean: norm={np.linalg.norm(cam_enroll_clean):.4f}")

    # CAM++ augmented enrollment
    cam_enroll_aug = campp.extract_augmented_embedding(enrollment_audio)
    print(f"  CAM++ augmented: norm={np.linalg.norm(cam_enroll_aug):.4f}")
    print(f"  CAM++ clean vs aug cosine: {cosine_similarity(cam_enroll_clean, cam_enroll_aug):.4f}")

    # ── Run all configurations ──
    configs = {
        "1_wespeaker_clean": (wespeaker, we_enroll_clean),
        "2_wespeaker_aug":   (wespeaker, we_enroll_aug),
        "3_campp_clean":     (campp, cam_enroll_clean),
        "4_campp_aug":       (campp, cam_enroll_aug),
    }

    all_results = {}
    all_metrics = {}

    for config_name, (encoder, dvector) in configs.items():
        print(f"\n{'─' * 50}")
        print(f"Config: {config_name}")
        print("─" * 50)

        result = run_pipeline(
            enrollment_dvector=dvector,
            input_audio=mixed_audio,
            denoiser=denoiser,
            speaker_encoder=encoder,
            threshold=THRESHOLD,
            pvad_interval=32,
        )

        # Save output audio
        write_audio(str(OUTPUT_DIR / f"{config_name}_output.wav"), result["output_audio"])

        # Compute metrics
        metrics = compute_metrics(result, labels, threshold=THRESHOLD)
        all_results[config_name] = result
        all_metrics[config_name] = metrics

        print(f"  Metrics:")
        for seg_name, m in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name
            marker = "TARGET" if is_hsuan else "INTERF"
            print(f"    [{marker}] {seg_name}: "
                  f"sim={m['mean_similarity']:.3f}±{m['std_similarity']:.3f}, "
                  f"target_ratio={m['target_ratio']:.1%}")

    # ── Plot comparison ──
    print("\n繪製對比圖...")
    plot_comparison(all_results, labels, OUTPUT_DIR / "comparison_chart.png", threshold=THRESHOLD)

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("SUMMARY — Scenario C (Alternating + White Noise SNR 10dB)")
    print("=" * 90)
    print(f"{'Config':<25} {'Segment':<20} {'Mean Sim':>10} {'Std':>8} {'Target%':>10}")
    print("-" * 90)

    for config_name, metrics in all_metrics.items():
        for seg_name, m in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name
            marker = "✓" if is_hsuan else "✗"
            print(f"  {marker} {config_name:<23} {seg_name:<20} "
                  f"{m['mean_similarity']:>10.3f} {m['std_similarity']:>8.3f} "
                  f"{m['target_ratio']:>9.1%}")

    # ── Aggregate stats ──
    print("\n" + "=" * 70)
    print("AGGREGATE — hsuan segments only")
    print("=" * 70)
    for config_name, metrics in all_metrics.items():
        hsuan_sims = []
        hsuan_ratios = []
        inter_sims = []
        inter_ratios = []
        for seg_name, m in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name
            if is_hsuan:
                hsuan_sims.append(m["mean_similarity"])
                hsuan_ratios.append(m["target_ratio"])
            else:
                inter_sims.append(m["mean_similarity"])
                inter_ratios.append(m["target_ratio"])
        avg_hsuan_sim = np.mean(hsuan_sims) if hsuan_sims else 0
        avg_hsuan_ratio = np.mean(hsuan_ratios) if hsuan_ratios else 0
        avg_inter_sim = np.mean(inter_sims) if inter_sims else 0
        avg_inter_ratio = np.mean(inter_ratios) if inter_ratios else 0
        print(f"  {config_name:<25}: "
              f"hsuan_sim={avg_hsuan_sim:.3f} target%={avg_hsuan_ratio:.1%} | "
              f"inter_sim={avg_inter_sim:.3f} false_pos%={avg_inter_ratio:.1%}")

    # ── Save report ──
    report = {
        "scenario": "C: Alternating + White Noise SNR 10dB",
        "threshold": THRESHOLD,
        "configs": all_metrics,
    }
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
