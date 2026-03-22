#!/usr/bin/env python3
"""
Personal VAD 測試 — 四場景驗證
===============================
用 MAT 資料集跑四場景，對比消融實驗 baseline。
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

from utils.audio import SAMPLE_RATE, read_audio, write_audio, frame_signal, overlap_add
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, _compute_fbank
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP
from utils.personal_vad import PersonalVAD

MAT_DIR = Path(os.path.expanduser("~/Desktop/VOICE/MAT"))
OUTPUT_DIR = PROJECT_DIR / "test_personal_vad"
OUTPUT_DIR.mkdir(exist_ok=True)

ENROLLMENT_CLEAN = MAT_DIR / "b6dbc0fc-1d57-4647-aa85-54f9bea08743_hsuan_7.wav"
HSUAN_SOURCE     = MAT_DIR / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"
INTERFERER_0911  = MAT_DIR / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"
INTERFERER_FEMH  = MAT_DIR / "ef81890f-791f-45c5-925a-2e7931a8f3c6_FEMH_7.wav"

PVAD_MODEL_PATH  = PROJECT_DIR / "models" / "personal_vad" / "personal_vad.onnx"


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)

def mix_at_snr(signal, noise, snr_db):
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return signal + noise * scale

def trim_or_pad(audio, length):
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])

def build_scenario_a(hsuan_audio, interferer_audio, seg_sec=2.5):
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

def build_scenario_b(hsuan_audio, interferer_audio, total_sec=8.0,
                     overlap_start_sec=2.5, overlap_end_sec=5.5):
    total_len = int(total_sec * SAMPLE_RATE)
    hsuan_seg = trim_or_pad(hsuan_audio, total_len)
    ov_s = int(overlap_start_sec * SAMPLE_RATE)
    ov_e = int(overlap_end_sec * SAMPLE_RATE)
    inter_seg = trim_or_pad(interferer_audio, ov_e - ov_s)
    scale = rms(hsuan_seg[ov_s:ov_e]) / (rms(inter_seg) + 1e-12)
    mixed = hsuan_seg.copy()
    mixed[ov_s:ov_e] += inter_seg * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * 0.95 / peak
    labels = [
        (0.0, overlap_start_sec, "hsuan_only"),
        (overlap_start_sec, overlap_end_sec, "hsuan+interferer"),
        (overlap_end_sec, total_sec, "hsuan_only"),
    ]
    return mixed, labels

def build_scenario_c(hsuan_audio, interferer_audio, seg_sec=2.5, noise_snr_db=10):
    mixed_clean, labels = build_scenario_a(hsuan_audio, interferer_audio, seg_sec=seg_sec)
    np.random.seed(42)
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak
    return mixed_noisy, labels


def run_personal_vad_pipeline(enrollment_path, input_audio, denoiser,
                               speaker_encoder, pvad_model_path):
    """用 Personal VAD 跑完整 pipeline。"""
    # 1. Enrollment d-vector
    enrollment_audio = read_audio(str(enrollment_path))
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)

    # 2. GTCRN denoise
    denoised_audio = denoiser.enhance(input_audio)

    # 3. Personal VAD on raw audio
    pvad = PersonalVAD(str(pvad_model_path), enrollment_dvector)
    gate = SoftGate(gain_floor=0.0, attack_ms=5.0, release_ms=30.0, hop=GTCRN_HOP)

    # 逐幀處理
    frames = frame_signal(denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = frames.shape[0]
    enhanced_frames = np.empty_like(frames)
    confidences = []
    is_targets = []

    for i in range(n_frames):
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(input_audio))
        raw_frame = input_audio[start:end]

        is_target, confidence = pvad.process_frame(raw_frame)
        confidences.append(confidence)
        is_targets.append(is_target)

        enhanced_frames[i] = gate.process(frames[i], is_target, confidence=confidence)

    output_audio = overlap_add(enhanced_frames, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    output_audio = output_audio[:len(input_audio)]
    peak = np.max(np.abs(output_audio))
    if peak > 0.99:
        output_audio = output_audio * 0.99 / peak

    return {
        "output_audio": output_audio,
        "denoised_audio": denoised_audio[:len(input_audio)],
        "similarities": np.array(confidences),  # target class probability
        "is_targets": np.array(is_targets),
        "n_frames": n_frames,
    }


def compute_metrics(result, labels, threshold=0.5):
    """計算各段指標。threshold=0.5 代表 target prob > 0.5 即判定。"""
    sims = result["similarities"]
    frame_times = np.arange(result["n_frames"]) * GTCRN_HOP / SAMPLE_RATE
    metrics = {}
    for (s, e, spk) in labels:
        mask = (frame_times >= s) & (frame_times < e)
        seg_sims = sims[mask]
        if len(seg_sims) > 0:
            metrics[f"{spk}_{s:.1f}-{e:.1f}s"] = {
                "mean_confidence": float(np.mean(seg_sims)),
                "std_confidence": float(np.std(seg_sims)),
                "target_ratio": float(result["is_targets"][mask].mean()),
                "n_frames": int(len(seg_sims)),
            }
    return metrics


def plot_timeline(result, labels, title, output_path):
    """繪製 Personal VAD timeline。"""
    confs = result["similarities"]
    n_frames = result["n_frames"]
    frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = {"hsuan": "#2ca02c", "interferer": "#d62728",
              "hsuan_only": "#2ca02c", "hsuan+interferer": "#ff7f0e"}

    # 1) Confidence
    ax1 = axes[0]
    ax1.plot(frame_times, confs, linewidth=0.8, color="steelblue", label="P(target)")
    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="decision boundary")
    for (s, e, spk) in labels:
        ax1.axvspan(s, e, alpha=0.15, color=colors.get(spk, "#999"))
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("Target Probability")
    ax1.set_title(f"{title} - Personal VAD (frame-level)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # 2) Target decision
    ax2 = axes[1]
    ax2.fill_between(frame_times, result["is_targets"].astype(float),
                     step="mid", alpha=0.5, color="green", label="is_target")
    for (s, e, spk) in labels:
        ax2.axvspan(s, e, alpha=0.1, color=colors.get(spk, "#999"))
    ax2.set_ylabel("Target Active")
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) Waveform
    ax3 = axes[2]
    t_out = np.arange(len(result["output_audio"])) / SAMPLE_RATE
    t_den = np.arange(len(result["denoised_audio"])) / SAMPLE_RATE
    ax3.plot(t_den, result["denoised_audio"], alpha=0.4, linewidth=0.3, color="gray", label="denoised")
    ax3.plot(t_out, result["output_audio"], alpha=0.7, linewidth=0.3, color="steelblue", label="output")
    for (s, e, spk) in labels:
        ax3.axvspan(s, e, alpha=0.1, color=colors.get(spk, "#999"))
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("Personal VAD Test - 4 Scenarios")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    campp_path = PROJECT_DIR / "models" / "campplus" / "campplus.onnx"
    speaker_encoder = SpeakerEncoder(str(campp_path))
    denoiser = GTCRNDenoiser()
    print(f"  Personal VAD model: {PVAD_MODEL_PATH}")
    print(f"  Speaker encoder: CAM++ (dim={speaker_encoder.embed_dim})")

    # Load audio
    print("\nLoading audio...")
    hsuan_audio = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))
    audio_femh = read_audio(str(INTERFERER_FEMH))

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

    all_metrics = {}

    for sc_id, sc_title, build_fn in scenarios:
        print(f"\n{'=' * 50}")
        print(f"Scenario {sc_title}")
        print("=" * 50)

        mixed, labels = build_fn()
        write_audio(str(OUTPUT_DIR / f"{sc_id}_mixed.wav"), mixed)

        t0 = time.time()
        result = run_personal_vad_pipeline(
            enrollment_path=ENROLLMENT_CLEAN,
            input_audio=mixed,
            denoiser=denoiser,
            speaker_encoder=speaker_encoder,
            pvad_model_path=PVAD_MODEL_PATH,
        )
        elapsed = time.time() - t0
        print(f"  Pipeline done: {elapsed:.1f}s")

        write_audio(str(OUTPUT_DIR / f"{sc_id}_output.wav"), result["output_audio"])
        write_audio(str(OUTPUT_DIR / f"{sc_id}_denoised.wav"), result["denoised_audio"])

        plot_timeline(result, labels, f"Scenario {sc_title}",
                      OUTPUT_DIR / f"{sc_id}_pvad.png")

        metrics = compute_metrics(result, labels)
        all_metrics[sc_id] = metrics

        for seg_name, m in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            marker = "[O] TARGET" if is_hsuan else "[X] INTERF"
            print(f"  {marker} [{seg_name}]: "
                  f"conf={m['mean_confidence']:.3f}+/-{m['std_confidence']:.3f}, "
                  f"target_ratio={m['target_ratio']:.1%}")

    # Save report
    report = {"model": "Personal VAD (LSTM 124K params)", "metrics": all_metrics}
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Summary comparison with ablation baseline
    print(f"\n{'=' * 70}")
    print("Personal VAD vs Ablation Baseline (CAM++ 0.5s window)")
    print(f"{'=' * 70}")
    print(f"{'Segment':35s} {'pVAD TR':>10s} {'PersonalVAD TR':>15s}")
    print("-" * 70)

    # Load ablation baseline
    ablation_path = PROJECT_DIR / "test_ablation" / "ablation_report.json"
    baseline_data = {}
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation = json.load(f)
        if "baseline" in ablation:
            baseline_data = ablation["baseline"].get("detailed_metrics", {})

    for sc_id in ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]:
        pvad_metrics = all_metrics.get(sc_id, {})
        base_metrics = baseline_data.get(sc_id, {})

        for seg_name in pvad_metrics:
            pvad_tr = pvad_metrics[seg_name]["target_ratio"]
            base_tr = base_metrics.get(seg_name, {}).get("target_ratio", float("nan"))
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            marker = "TGT" if is_hsuan else "INT"
            print(f"  [{marker}] {sc_id}/{seg_name:25s} {base_tr:9.1%} {pvad_tr:14.1%}")

    print(f"\nAll outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
