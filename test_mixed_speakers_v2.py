#!/usr/bin/env python3
"""
混合說話者測試 v2 — threshold=0.35
===================================
與 v1 (threshold=0.25) 相同場景，只改 threshold，輸出到 test_mixed_v2/。
結尾附 v1 vs v2 比較。
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
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP

MODELS_DIR = PROJECT_DIR / "models"
MAT_DIR = Path("/sessions/quirky-zen-rubin/mnt/MAT")
OUTPUT_DIR = PROJECT_DIR / "test_mixed_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

V1_REPORT = PROJECT_DIR / "test_mixed" / "test_report.json"

THRESHOLD = 0.35  # v1 was 0.25

# ── 音頻檔案 ────────────────────────────────────────
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


# ── 場景構造 ─────────────────────────────────────────

def build_scenario_a(hsuan_audio, interferer_audio, seg_sec=2.5):
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
    return np.concatenate(segments), labels


def build_scenario_b(hsuan_audio, interferer_audio, total_sec=8.0,
                     overlap_start_sec=2.5, overlap_end_sec=5.5):
    total_len = int(total_sec * SAMPLE_RATE)
    hsuan_seg = trim_or_pad(hsuan_audio, total_len)
    ov_s, ov_e = int(overlap_start_sec * SAMPLE_RATE), int(overlap_end_sec * SAMPLE_RATE)
    inter_seg = trim_or_pad(interferer_audio, ov_e - ov_s)
    scale = rms(hsuan_seg[ov_s:ov_e]) / (rms(inter_seg) + 1e-12)
    mixed = hsuan_seg.copy()
    mixed[ov_s:ov_e] += inter_seg * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * 0.95 / peak
    labels = [(0.0, overlap_start_sec, "hsuan_only"),
              (overlap_start_sec, overlap_end_sec, "hsuan+interferer"),
              (overlap_end_sec, total_sec, "hsuan_only")]
    return mixed, labels


def build_scenario_c(hsuan_audio, interferer_audio, seg_sec=2.5, noise_snr_db=10):
    mixed_clean, labels = build_scenario_a(hsuan_audio, interferer_audio, seg_sec=seg_sec)
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak
    return mixed_noisy, labels


# ── Pipeline 執行 ────────────────────────────────────

def run_pipeline_with_tracking(enrollment_path, input_audio, denoiser, speaker_encoder,
                               threshold=0.35, gain_floor=0.05, pvad_interval=32):
    enrollment_audio = read_audio(str(enrollment_path))
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)

    # Step 1: GTCRN denoise
    denoised_audio = denoiser.enhance(input_audio)

    # Step 2: pVAD on denoised audio + gating
    pvad = CachedPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        threshold=threshold,
    )
    gate = SoftGate(gain_floor=gain_floor, attack_ms=5.0, release_ms=50.0, hop=GTCRN_HOP)

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


# ── 視覺化 ───────────────────────────────────────────

def plot_pvad_timeline(result, labels, title, output_path, threshold=0.35):
    sims = result["similarities"]
    n_frames = result["n_frames"]
    frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(frame_times, sims, linewidth=0.8, color="steelblue", label="pVAD similarity")
    ax1.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7,
                label=f"threshold={threshold}")
    # Also show old threshold for comparison
    ax1.axhline(y=0.25, color="orange", linestyle=":", linewidth=1, alpha=0.5,
                label="old threshold=0.25")

    colors = {"hsuan": "#2ca02c", "interferer": "#d62728",
              "hsuan_only": "#2ca02c", "hsuan+interferer": "#ff7f0e"}
    for (s, e, spk) in labels:
        ax1.axvspan(s, e, alpha=0.15, color=colors.get(spk, "#999"))
    for i in range(1, len(labels)):
        ax1.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)

    handles, leg_labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title(f"{title} — pVAD Similarity (threshold=0.35)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(frame_times, result["is_targets"].astype(float),
                     step="mid", alpha=0.5, color="green", label="is_target=True")
    for (s, e, spk) in labels:
        ax2.axvspan(s, e, alpha=0.1, color=colors.get(spk, "#999"))
    for i in range(1, len(labels)):
        ax2.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Target Active")
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    input_t = np.arange(len(result["denoised_audio"])) / SAMPLE_RATE
    output_t = np.arange(len(result["output_audio"])) / SAMPLE_RATE
    ax3.plot(input_t, result["denoised_audio"], alpha=0.4, linewidth=0.3, color="gray",
             label="denoised (input)")
    ax3.plot(output_t, result["output_audio"], alpha=0.7, linewidth=0.3, color="steelblue",
             label="gated (output)")
    for (s, e, spk) in labels:
        ax3.axvspan(s, e, alpha=0.1, color=colors.get(spk, "#999"))
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  chart saved: {output_path}")


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


# ── v1 vs v2 比較 ────────────────────────────────────

def compare_v1_v2(v1_metrics, v2_metrics):
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("v1 (threshold=0.25) vs v2 (threshold=0.35) 比較")
    print("=" * 80)
    for scenario in sorted(set(list(v1_metrics.keys()) + list(v2_metrics.keys()))):
        print(f"\n  {scenario}:")
        v1_segs = v1_metrics.get(scenario, {})
        v2_segs = v2_metrics.get(scenario, {})
        all_seg_keys = sorted(set(list(v1_segs.keys()) + list(v2_segs.keys())))
        for seg_key in all_seg_keys:
            is_hsuan = "hsuan" in seg_key and "interferer" not in seg_key and "+" not in seg_key
            marker = "TARGET" if is_hsuan else "INTERF"
            v1s = v1_segs.get(seg_key, {})
            v2s = v2_segs.get(seg_key, {})
            v1_sim = v1s.get("mean_similarity", float("nan"))
            v2_sim = v2s.get("mean_similarity", float("nan"))
            v1_tr = v1s.get("target_ratio", float("nan"))
            v2_tr = v2s.get("target_ratio", float("nan"))
            delta_tr = v2_tr - v1_tr if not (np.isnan(v1_tr) or np.isnan(v2_tr)) else float("nan")
            print(f"    [{marker}] {seg_key}")
            print(f"      sim: {v1_sim:.3f} -> {v2_sim:.3f}  |  "
                  f"target_ratio: {v1_tr:.1%} -> {v2_tr:.1%} ({delta_tr:+.1%})")


# ── 主流程 ───────────────────────────────────────────

def main():
    np.random.seed(42)  # reproducible noise for scenario C

    print("=" * 60)
    print("混合說話者 GTCRN + pVAD 測試 v2 (threshold=0.35)")
    print("=" * 60)

    print("\n載入模型...")
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    assert ecapa_path.exists(), f"找不到 speaker encoder: {ecapa_path}"
    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    denoiser = GTCRNDenoiser()
    print("  done")

    print("\n載入語音素材...")
    hsuan_audio = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))
    audio_femh = read_audio(str(INTERFERER_FEMH))
    print(f"  hsuan: {len(hsuan_audio)/SAMPLE_RATE:.1f}s")
    print(f"  0911: {len(audio_0911)/SAMPLE_RATE:.1f}s")
    print(f"  FEMH: {len(audio_femh)/SAMPLE_RATE:.1f}s")

    all_metrics = {}

    scenarios = [
        ("scenario_a", "A: Alternating (hsuan <-> 0911)",
         lambda: build_scenario_a(hsuan_audio, audio_0911)),
        ("scenario_b", "B: Partial Overlap (hsuan + 0911 @ 0dB)",
         lambda: build_scenario_b(hsuan_audio, audio_0911)),
        ("scenario_c", "C: Alternating + White Noise SNR 10dB",
         lambda: build_scenario_c(hsuan_audio, audio_0911)),
        ("scenario_d", "D: Alternating (hsuan <-> FEMH)",
         lambda: build_scenario_a(hsuan_audio, audio_femh)),
    ]

    for sc_id, sc_title, build_fn in scenarios:
        print(f"\n{'=' * 50}")
        print(f"場景 {sc_title}")
        print("=" * 50)

        mixed, labels = build_fn()
        write_audio(str(OUTPUT_DIR / f"{sc_id}_mixed.wav"), mixed)
        print(f"  mixed: {len(mixed)/SAMPLE_RATE:.1f}s, segments: {len(labels)}")

        result = run_pipeline_with_tracking(
            HSUAN_ENROLLMENT, mixed, denoiser, speaker_encoder,
            threshold=THRESHOLD, pvad_interval=32
        )
        write_audio(str(OUTPUT_DIR / f"{sc_id}_output.wav"), result["output_audio"])
        write_audio(str(OUTPUT_DIR / f"{sc_id}_denoised.wav"), result["denoised_audio"])
        plot_pvad_timeline(result, labels, f"Scenario {sc_title}",
                           OUTPUT_DIR / f"{sc_id}_pvad.png", threshold=THRESHOLD)
        metrics = compute_metrics(result, labels, threshold=THRESHOLD)
        all_metrics[sc_id] = metrics
        print(f"  Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    # Save report
    report_path = OUTPUT_DIR / "test_report_v2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nv2 report saved: {report_path}")

    # Summary
    print("\n" + "=" * 60)
    print("v2 測試摘要 (threshold=0.35)")
    print("=" * 60)
    for sc_name, metrics in all_metrics.items():
        print(f"\n{sc_name}:")
        for seg_name, m in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            marker = "v TARGET" if is_hsuan else "x INTERF"
            print(f"  {marker} [{seg_name}]: "
                  f"sim={m['mean_similarity']:.3f}+-{m['std_similarity']:.3f}, "
                  f"target_ratio={m['target_ratio']:.1%}")

    # v1 vs v2 comparison
    if V1_REPORT.exists():
        with open(V1_REPORT, "r") as f:
            v1_metrics = json.load(f)
        compare_v1_v2(v1_metrics, all_metrics)

        # Save comparison as JSON too
        comparison = {}
        for sc in sorted(set(list(v1_metrics.keys()) + list(all_metrics.keys()))):
            comparison[sc] = {}
            v1s = v1_metrics.get(sc, {})
            v2s = all_metrics.get(sc, {})
            for seg in sorted(set(list(v1s.keys()) + list(v2s.keys()))):
                comparison[sc][seg] = {
                    "v1_threshold_0.25": v1s.get(seg, {}),
                    "v2_threshold_0.35": v2s.get(seg, {}),
                }
        comp_path = OUTPUT_DIR / "v1_vs_v2_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nComparison saved: {comp_path}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
