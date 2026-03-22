#!/usr/bin/env python3
"""
混合說話者測試腳本
==================
構造多場景混合音頻，跑 GTCRN + CachedPVAD pipeline，
生成 pVAD similarity 時間軸曲線並輸出 gated 音檔。

場景：
  A: 交替說話 (hsuan → interferer → hsuan)
  B: 部分重疊 (hsuan 持續，interferer 中間疊加 0dB)
  C: 交替 + 白噪音 SNR 10dB
  D: 同上但用 FEMH 作為干擾者
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 加入 project path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE, read_audio, write_audio
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP

MODELS_DIR = PROJECT_DIR / "models"
MAT_DIR = Path("/sessions/adoring-modest-franklin/mnt/MAT")
OUTPUT_DIR = PROJECT_DIR / "test_mixed"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 音頻檔案選擇 ────────────────────────────────────────
# hsuan: 用一段作 enrollment，另一段作為目標語音素材
HSUAN_ENROLLMENT = MAT_DIR / "275eaceb-2387-4f9e-aef5-1e9996b8f024_hsuan_7.wav"  # 17.7s
HSUAN_SOURCE     = MAT_DIR / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"  # 17.8s
INTERFERER_0911  = MAT_DIR / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"  # 18.3s
INTERFERER_FEMH  = MAT_DIR / "ef81890f-791f-45c5-925a-2e7931a8f3c6_FEMH_7.wav"  # 15.2s


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_at_snr(signal, noise, snr_db):
    """將 noise 縮放到相對 signal 的指定 SNR，然後相加。"""
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return signal + noise * scale


def trim_or_pad(audio, length):
    """裁剪或零填充到指定長度。"""
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])


# ── 場景構造 ─────────────────────────────────────────────

def build_scenario_a(hsuan_audio, interferer_audio, seg_sec=2.5):
    """
    場景 A：交替說話
    hsuan(seg_sec) → interferer(seg_sec) → hsuan(seg_sec) → interferer(seg_sec) → hsuan(seg_sec)
    回傳 (mixed_audio, speaker_labels)
    speaker_labels: list of (start_sec, end_sec, speaker_name)
    """
    seg_len = int(seg_sec * SAMPLE_RATE)
    segments = []
    labels = []
    t = 0.0

    # 從各自音頻中取片段
    h_offset = 0
    i_offset = 0

    for idx in range(5):
        if idx % 2 == 0:  # hsuan
            seg = hsuan_audio[h_offset:h_offset + seg_len]
            seg = trim_or_pad(seg, seg_len)
            segments.append(seg)
            labels.append((t, t + seg_sec, "hsuan"))
            h_offset += seg_len
        else:  # interferer
            seg = interferer_audio[i_offset:i_offset + seg_len]
            seg = trim_or_pad(seg, seg_len)
            segments.append(seg)
            labels.append((t, t + seg_sec, "interferer"))
            i_offset += seg_len
        t += seg_sec

    mixed = np.concatenate(segments)
    return mixed, labels


def build_scenario_b(hsuan_audio, interferer_audio, total_sec=8.0,
                     overlap_start_sec=2.5, overlap_end_sec=5.5):
    """
    場景 B：部分重疊
    hsuan 持續 total_sec，interferer 在 [overlap_start, overlap_end] 疊加 (SNR 0dB)
    """
    total_len = int(total_sec * SAMPLE_RATE)
    hsuan_seg = trim_or_pad(hsuan_audio, total_len)

    overlap_start = int(overlap_start_sec * SAMPLE_RATE)
    overlap_end = int(overlap_end_sec * SAMPLE_RATE)
    overlap_len = overlap_end - overlap_start

    interferer_seg = trim_or_pad(interferer_audio, overlap_len)

    # 計算 SNR 0dB 的縮放
    h_overlap = hsuan_seg[overlap_start:overlap_end]
    scale = rms(h_overlap) / (rms(interferer_seg) + 1e-12)
    # SNR=0dB 表示 scale=1 (相同 RMS)
    interferer_scaled = interferer_seg * scale

    mixed = hsuan_seg.copy()
    mixed[overlap_start:overlap_end] += interferer_scaled

    # 避免 clipping
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
    """
    場景 C：交替說話 + 白噪音 (SNR 10dB)
    """
    mixed_clean, labels = build_scenario_a(hsuan_audio, interferer_audio, seg_sec=seg_sec)

    # 加白噪音
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)

    # 避免 clipping
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak

    return mixed_noisy, labels


# ── Pipeline 執行 ────────────────────────────────────────

def run_pipeline_with_tracking(enrollment_path, input_audio, denoiser, speaker_encoder,
                               threshold=0.25, gain_floor=0.05, pvad_interval=32):
    """
    跑完整 pipeline 並回傳逐幀 similarity + gated 音頻。
    """
    # Enrollment
    enrollment_audio = read_audio(str(enrollment_path))
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)

    # GTCRN 降噪
    denoised_audio = denoiser.enhance(input_audio)

    # CachedPVAD + gating
    pvad = CachedPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        threshold=threshold,
    )
    gate = SoftGate(gain_floor=gain_floor, attack_ms=5.0, release_ms=50.0, hop=GTCRN_HOP)

    from utils.audio import frame_signal, overlap_add

    frames = frame_signal(denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = frames.shape[0]

    enhanced_frames = np.empty_like(frames)
    similarities = []
    is_targets = []

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


# ── 視覺化 ───────────────────────────────────────────────

def plot_pvad_timeline(result, labels, title, output_path, threshold=0.25):
    """
    畫 pVAD similarity 時間軸曲線，標註說話者切換點。
    """
    sims = result["similarities"]
    n_frames = result["n_frames"]
    frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # --- 子圖1：pVAD similarity ---
    ax1 = axes[0]
    ax1.plot(frame_times, sims, linewidth=0.8, color="steelblue", label="pVAD similarity")
    ax1.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"threshold={threshold}")

    # 標註說話者區間
    colors = {"hsuan": "#2ca02c", "interferer": "#d62728",
              "hsuan_only": "#2ca02c", "hsuan+interferer": "#ff7f0e"}
    for (start_s, end_s, spk) in labels:
        color = colors.get(spk, "#999999")
        ax1.axvspan(start_s, end_s, alpha=0.15, color=color, label=spk)

    # 標註切換點
    for i in range(1, len(labels)):
        switch_t = labels[i][0]
        ax1.axvline(x=switch_t, color="black", linestyle=":", linewidth=1, alpha=0.5)

    # 去重 legend
    handles, leg_labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title(f"{title} — pVAD Similarity")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # --- 子圖2：is_target 判定 ---
    ax2 = axes[1]
    ax2.fill_between(frame_times, result["is_targets"].astype(float),
                     step="mid", alpha=0.5, color="green", label="is_target=True")
    for (start_s, end_s, spk) in labels:
        color = colors.get(spk, "#999999")
        ax2.axvspan(start_s, end_s, alpha=0.1, color=color)
    for i in range(1, len(labels)):
        ax2.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Target Active")
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 子圖3：波形對比（輸入 vs gated 輸出）---
    ax3 = axes[2]
    input_t = np.arange(len(result["denoised_audio"])) / SAMPLE_RATE
    output_t = np.arange(len(result["output_audio"])) / SAMPLE_RATE
    ax3.plot(input_t, result["denoised_audio"], alpha=0.4, linewidth=0.3, color="gray", label="denoised (input)")
    ax3.plot(output_t, result["output_audio"], alpha=0.7, linewidth=0.3, color="steelblue", label="gated (output)")
    for (start_s, end_s, spk) in labels:
        color = colors.get(spk, "#999999")
        ax3.axvspan(start_s, end_s, alpha=0.1, color=color)
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  圖表已存: {output_path}")


def compute_metrics(result, labels, threshold=0.25):
    """計算各區段的平均 similarity 和 target 比率。"""
    sims = result["similarities"]
    frame_times = np.arange(result["n_frames"]) * GTCRN_HOP / SAMPLE_RATE

    metrics = {}
    for (start_s, end_s, spk) in labels:
        mask = (frame_times >= start_s) & (frame_times < end_s)
        seg_sims = sims[mask]
        if len(seg_sims) > 0:
            metrics[f"{spk}_{start_s:.1f}-{end_s:.1f}s"] = {
                "mean_similarity": float(np.mean(seg_sims)),
                "std_similarity": float(np.std(seg_sims)),
                "target_ratio": float((seg_sims > threshold).mean()),
                "n_frames": int(len(seg_sims)),
            }
    return metrics


# ── 主流程 ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("混合說話者 GTCRN + pVAD 測試")
    print("=" * 60)

    # 載入模型（只載入一次）
    print("\n載入模型...")
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    assert ecapa_path.exists(), f"找不到 speaker encoder: {ecapa_path}"

    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    denoiser = GTCRNDenoiser()
    print("  模型載入完成")

    # 載入語音素材
    print("\n載入語音素材...")
    hsuan_audio = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))
    audio_femh = read_audio(str(INTERFERER_FEMH))
    print(f"  hsuan: {len(hsuan_audio)/SAMPLE_RATE:.1f}s")
    print(f"  0911636193: {len(audio_0911)/SAMPLE_RATE:.1f}s")
    print(f"  FEMH: {len(audio_femh)/SAMPLE_RATE:.1f}s")

    all_metrics = {}

    # ── 場景 A：交替說話 ──────────────────────────────
    print("\n" + "=" * 50)
    print("場景 A：交替說話 (hsuan ↔ 0911636193)")
    print("=" * 50)
    mixed_a, labels_a = build_scenario_a(hsuan_audio, audio_0911, seg_sec=2.5)
    write_audio(str(OUTPUT_DIR / "scenario_a_mixed.wav"), mixed_a)
    print(f"  混合音頻: {len(mixed_a)/SAMPLE_RATE:.1f}s")
    print(f"  區段: {labels_a}")

    result_a = run_pipeline_with_tracking(
        HSUAN_ENROLLMENT, mixed_a, denoiser, speaker_encoder,
        threshold=0.25, pvad_interval=32
    )
    write_audio(str(OUTPUT_DIR / "scenario_a_output.wav"), result_a["output_audio"])
    write_audio(str(OUTPUT_DIR / "scenario_a_denoised.wav"), result_a["denoised_audio"])
    plot_pvad_timeline(result_a, labels_a,
                       "Scenario A: Alternating Speakers (hsuan ↔ 0911)",
                       OUTPUT_DIR / "scenario_a_pvad.png")
    metrics_a = compute_metrics(result_a, labels_a)
    all_metrics["scenario_a"] = metrics_a
    print(f"  Metrics: {json.dumps(metrics_a, indent=2, ensure_ascii=False)}")

    # ── 場景 B：部分重疊 ──────────────────────────────
    print("\n" + "=" * 50)
    print("場景 B：部分重疊 (hsuan 持續 + 0911 中間疊加 0dB)")
    print("=" * 50)
    mixed_b, labels_b = build_scenario_b(hsuan_audio, audio_0911,
                                         total_sec=8.0,
                                         overlap_start_sec=2.5,
                                         overlap_end_sec=5.5)
    write_audio(str(OUTPUT_DIR / "scenario_b_mixed.wav"), mixed_b)
    print(f"  混合音頻: {len(mixed_b)/SAMPLE_RATE:.1f}s")

    result_b = run_pipeline_with_tracking(
        HSUAN_ENROLLMENT, mixed_b, denoiser, speaker_encoder,
        threshold=0.25, pvad_interval=32
    )
    write_audio(str(OUTPUT_DIR / "scenario_b_output.wav"), result_b["output_audio"])
    write_audio(str(OUTPUT_DIR / "scenario_b_denoised.wav"), result_b["denoised_audio"])
    plot_pvad_timeline(result_b, labels_b,
                       "Scenario B: Partial Overlap (hsuan + 0911 @ 0dB)",
                       OUTPUT_DIR / "scenario_b_pvad.png")
    metrics_b = compute_metrics(result_b, labels_b)
    all_metrics["scenario_b"] = metrics_b
    print(f"  Metrics: {json.dumps(metrics_b, indent=2, ensure_ascii=False)}")

    # ── 場景 C：交替 + 白噪音 ────────────────────────
    print("\n" + "=" * 50)
    print("場景 C：交替說話 + 白噪音 SNR 10dB")
    print("=" * 50)
    mixed_c, labels_c = build_scenario_c(hsuan_audio, audio_0911,
                                         seg_sec=2.5, noise_snr_db=10)
    write_audio(str(OUTPUT_DIR / "scenario_c_mixed.wav"), mixed_c)
    print(f"  混合音頻: {len(mixed_c)/SAMPLE_RATE:.1f}s")

    result_c = run_pipeline_with_tracking(
        HSUAN_ENROLLMENT, mixed_c, denoiser, speaker_encoder,
        threshold=0.25, pvad_interval=32
    )
    write_audio(str(OUTPUT_DIR / "scenario_c_output.wav"), result_c["output_audio"])
    write_audio(str(OUTPUT_DIR / "scenario_c_denoised.wav"), result_c["denoised_audio"])
    plot_pvad_timeline(result_c, labels_c,
                       "Scenario C: Alternating + White Noise (SNR 10dB)",
                       OUTPUT_DIR / "scenario_c_pvad.png")
    metrics_c = compute_metrics(result_c, labels_c)
    all_metrics["scenario_c"] = metrics_c
    print(f"  Metrics: {json.dumps(metrics_c, indent=2, ensure_ascii=False)}")

    # ── 場景 D：FEMH 作為干擾者 ──────────────────────
    print("\n" + "=" * 50)
    print("場景 D：交替說話 (hsuan ↔ FEMH)")
    print("=" * 50)
    mixed_d, labels_d = build_scenario_a(hsuan_audio, audio_femh, seg_sec=2.5)
    write_audio(str(OUTPUT_DIR / "scenario_d_mixed.wav"), mixed_d)

    result_d = run_pipeline_with_tracking(
        HSUAN_ENROLLMENT, mixed_d, denoiser, speaker_encoder,
        threshold=0.25, pvad_interval=32
    )
    write_audio(str(OUTPUT_DIR / "scenario_d_output.wav"), result_d["output_audio"])
    write_audio(str(OUTPUT_DIR / "scenario_d_denoised.wav"), result_d["denoised_audio"])
    plot_pvad_timeline(result_d, labels_d,
                       "Scenario D: Alternating Speakers (hsuan ↔ FEMH)",
                       OUTPUT_DIR / "scenario_d_pvad.png")
    metrics_d = compute_metrics(result_d, labels_d)
    all_metrics["scenario_d"] = metrics_d
    print(f"  Metrics: {json.dumps(metrics_d, indent=2, ensure_ascii=False)}")

    # ── 總結報告 ──────────────────────────────────────
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n完整報告已存: {report_path}")

    # 印出摘要
    print("\n" + "=" * 60)
    print("測試摘要")
    print("=" * 60)
    for scenario_name, metrics in all_metrics.items():
        print(f"\n{scenario_name}:")
        for seg_name, seg_metrics in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            marker = "✓ 目標" if is_hsuan else "✗ 非目標"
            print(f"  {marker} [{seg_name}]: "
                  f"sim={seg_metrics['mean_similarity']:.3f}±{seg_metrics['std_similarity']:.3f}, "
                  f"target_ratio={seg_metrics['target_ratio']:.1%}")

    print(f"\n所有輸出已存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
