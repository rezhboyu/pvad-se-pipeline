#!/usr/bin/env python3
"""
v3 CAM++ + EMA: parallel vs serial vs hybrid comparison
========================================================
Config 1: parallel — pVAD on raw audio
Config 2: serial   — pVAD on GTCRN-denoised audio
Config 3: hybrid   — estimate SNR, auto-switch

Scenarios A/C/D (B omitted for brevity).
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE, read_audio, write_audio
from utils.speaker_encoder import SpeakerEncoder
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_HOP
from pipeline_parallel import run_parallel_pipeline_from_audio

MODELS_DIR = PROJECT_DIR / "models"
MAT_DIR = Path(os.path.expanduser("~/Desktop/VOICE/MAT"))
OUTPUT_DIR = PROJECT_DIR / "test_v3_compare"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.25
EMA_ALPHA = 0.6
PVAD_WINDOW = 0.5

# Audio files
HSUAN_ENROLLMENT = MAT_DIR / "275eaceb-2387-4f9e-aef5-1e9996b8f024_hsuan_7.wav"
HSUAN_SOURCE     = MAT_DIR / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"
INTERFERER_0911  = MAT_DIR / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"
INTERFERER_FEMH  = MAT_DIR / "ef81890f-791f-45c5-925a-2e7931a8f3c6_FEMH_7.wav"


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)

def mix_at_snr(signal, noise, snr_db):
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return signal + noise * scale

def trim_or_pad(audio, length):
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])

def estimate_snr(audio, frame_len=4096):
    """Simple SNR estimate: ratio of top-20% RMS to bottom-20% RMS."""
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return 30.0  # assume clean
    rms_vals = []
    for i in range(n_frames):
        chunk = audio[i * frame_len:(i + 1) * frame_len]
        rms_vals.append(np.sqrt(np.mean(chunk ** 2) + 1e-12))
    rms_vals = np.sort(rms_vals)
    n20 = max(1, n_frames // 5)
    noise_floor = np.mean(rms_vals[:n20])
    signal_level = np.mean(rms_vals[-n20:])
    if noise_floor < 1e-10:
        return 40.0
    return 20 * np.log10(signal_level / noise_floor)


def build_alternating(hsuan_audio, interferer_audio, seg_sec=2.5):
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
    return np.concatenate(segments), labels


def build_noisy(hsuan_audio, interferer_audio, snr_db=10):
    mixed_clean, labels = build_alternating(hsuan_audio, interferer_audio)
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed = mix_at_snr(mixed_clean, noise, snr_db)
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * 0.95 / peak
    return mixed, labels


def compute_metrics(result, labels, threshold):
    sims = result["similarities"]
    frame_times = np.arange(result["n_frames"]) * GTCRN_HOP / SAMPLE_RATE
    metrics = {}
    for (s, e, spk) in labels:
        mask = (frame_times >= s) & (frame_times < e)
        seg_sims = sims[mask]
        if len(seg_sims) > 0:
            metrics[f"{spk}_{s:.1f}-{e:.1f}s"] = {
                "mean_sim": round(float(np.mean(seg_sims)), 3),
                "target_ratio": round(float((seg_sims > threshold).mean()), 3),
            }
    return metrics


def run_config(name, mixed, speaker_encoder, denoiser, pvad_source):
    t0 = time.time()
    result = run_parallel_pipeline_from_audio(
        enrollment_path=HSUAN_ENROLLMENT,
        input_audio=mixed,
        denoiser=denoiser,
        speaker_encoder=speaker_encoder,
        threshold=THRESHOLD,
        pvad_interval=32,
        pvad_window_sec=PVAD_WINDOW,
        ema_alpha=EMA_ALPHA,
        use_augmented_enrollment=False,
        pvad_source=pvad_source,
    )
    elapsed = time.time() - t0
    print(f"  {name}: {elapsed:.1f}s")
    return result


def plot_comparison(results_dict, labels, title, output_path):
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = ["steelblue", "darkorange", "forestgreen"]
    spk_colors = {"hsuan": "#2ca02c", "interferer": "#d62728"}

    for idx, (cfg_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        sims = result["similarities"]
        ft = np.arange(result["n_frames"]) * GTCRN_HOP / SAMPLE_RATE
        ax.plot(ft, sims, linewidth=0.8, color=colors[idx % len(colors)])
        ax.axhline(y=THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.7)
        for (s, e, spk) in labels:
            ax.axvspan(s, e, alpha=0.15, color=spk_colors.get(spk, "#999"))
        for i in range(1, len(labels)):
            ax.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_ylabel("Similarity")
        ax.set_title(f"{cfg_name}", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.1, 0.8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def main():
    np.random.seed(42)

    print("=" * 60)
    print("v3 CAM++ + EMA: parallel vs serial vs hybrid")
    print(f"  threshold={THRESHOLD}, window={PVAD_WINDOW}s, ema_alpha={EMA_ALPHA}")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    campp_path = MODELS_DIR / "campplus" / "campplus.onnx"
    assert campp_path.exists(), f"CAM++ not found: {campp_path}"
    speaker_encoder = SpeakerEncoder(str(campp_path))
    denoiser = GTCRNDenoiser()
    print(f"  CAM++ embed_dim={speaker_encoder.embed_dim}")

    # Load audio
    print("\nLoading audio...")
    hsuan = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))
    audio_femh = read_audio(str(INTERFERER_FEMH))

    scenarios = {
        "A_clean": ("Alternating (hsuan <-> 0911)", lambda: build_alternating(hsuan, audio_0911)),
        "C_noisy": ("Alternating + White Noise 10dB", lambda: build_noisy(hsuan, audio_0911, 10)),
        "D_clean": ("Alternating (hsuan <-> FEMH)", lambda: build_alternating(hsuan, audio_femh)),
    }

    all_results = {}

    for sc_id, (sc_title, build_fn) in scenarios.items():
        print(f"\n{'=' * 50}")
        print(f"Scenario {sc_id}: {sc_title}")
        print("=" * 50)

        mixed, labels = build_fn()
        snr_est = estimate_snr(mixed)
        print(f"  mixed: {len(mixed)/SAMPLE_RATE:.1f}s, estimated SNR: {snr_est:.1f}dB")
        write_audio(str(OUTPUT_DIR / f"{sc_id}_mixed.wav"), mixed)

        # Config 1: parallel (pVAD on raw)
        r_par = run_config("parallel (raw)", mixed, speaker_encoder, denoiser, "raw")

        # Config 2: serial (pVAD on denoised)
        r_ser = run_config("serial (denoised)", mixed, speaker_encoder, denoiser, "denoised")

        # Config 3: hybrid (auto-select based on SNR)
        SNR_THRESHOLD = 20.0
        if snr_est >= SNR_THRESHOLD:
            hybrid_source = "raw"
            hybrid_label = f"hybrid -> raw (SNR={snr_est:.0f}dB >= {SNR_THRESHOLD})"
        else:
            hybrid_source = "denoised"
            hybrid_label = f"hybrid -> denoised (SNR={snr_est:.0f}dB < {SNR_THRESHOLD})"
        r_hyb = run_config(hybrid_label, mixed, speaker_encoder, denoiser, hybrid_source)

        # Metrics
        configs = {
            "parallel": r_par,
            "serial": r_ser,
            "hybrid": r_hyb,
        }
        sc_metrics = {}
        for cfg_name, result in configs.items():
            m = compute_metrics(result, labels, THRESHOLD)
            sc_metrics[cfg_name] = m
            write_audio(str(OUTPUT_DIR / f"{sc_id}_{cfg_name}_output.wav"), result["output_audio"])

        all_results[sc_id] = sc_metrics

        # Print comparison table
        print(f"\n  {'Segment':<25} {'parallel':>12} {'serial':>12} {'hybrid':>12}")
        print(f"  {'-'*61}")
        all_segs = sorted(set().union(*(m.keys() for m in sc_metrics.values())))
        for seg in all_segs:
            is_hsuan = "hsuan" in seg and "interferer" not in seg
            marker = "T" if is_hsuan else "I"
            vals = []
            for cfg in ["parallel", "serial", "hybrid"]:
                m = sc_metrics[cfg].get(seg, {})
                tr = m.get("target_ratio", 0)
                sim = m.get("mean_sim", 0)
                vals.append(f"{sim:.3f}/{tr:.0%}")
            print(f"  [{marker}] {seg:<22} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

        # Plot
        plot_comparison(
            {"parallel (raw)": r_par, "serial (denoised)": r_ser, f"hybrid ({hybrid_source})": r_hyb},
            labels,
            f"Scenario {sc_id}: {sc_title}",
            OUTPUT_DIR / f"{sc_id}_comparison.png",
        )
        print(f"  chart: {OUTPUT_DIR / f'{sc_id}_comparison.png'}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: best config per scenario")
    print("=" * 70)
    for sc_id, sc_metrics in all_results.items():
        print(f"\n  {sc_id}:")
        for cfg_name, metrics in sc_metrics.items():
            hsuan_ratios = [m["target_ratio"] for k, m in metrics.items()
                           if "hsuan" in k and "interferer" not in k]
            inter_ratios = [m["target_ratio"] for k, m in metrics.items()
                           if "interferer" in k]
            h_avg = np.mean(hsuan_ratios) if hsuan_ratios else 0
            i_avg = np.mean(inter_ratios) if inter_ratios else 0
            score = h_avg - i_avg  # higher = better separation
            print(f"    {cfg_name:12s}: hsuan={h_avg:.1%}  interf={i_avg:.1%}  gap={score:+.1%}")

    # Save report
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {report_path}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
