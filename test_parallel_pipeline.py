#!/usr/bin/env python3
"""
並行管線測試 — 四場景 + v1 vs v2 對比
=========================================
測試新的並行架構（pVAD 從 raw audio 提取 embedding）。

場景：
  A: hsuan ↔ 0911636193 交替說話
  B: hsuan + 0911636193 部分重疊
  C: 交替 + 白噪 SNR 10dB（串行版崩潰的場景）
  D: hsuan ↔ FEMH 交替說話

輸出（存到 test_parallel/）：
  - 每場景：mixed.wav, denoised.wav, output.wav
  - pVAD similarity 時間軸曲線圖
  - 場景 C 的 v1（串行）vs v2（並行）對比圖
  - 測試報告 JSON

用法:
    python test_parallel_pipeline.py
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
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP
from pipeline_parallel import run_parallel_pipeline_from_audio

MODELS_DIR = PROJECT_DIR / "models"

# ── 自動偵測 MAT 目錄 ────────────────────────────────
# 支援多個可能的掛載路徑
_MAT_CANDIDATES = [
    Path(os.path.expanduser("~/Desktop/VOICE/MAT")),
]
# Cowork VM 可能的掛載路徑
if Path("/sessions").exists():
    _MAT_CANDIDATES.extend(Path("/sessions").glob("*/mnt/MAT"))
MAT_DIR = None
for _p in _MAT_CANDIDATES:
    if _p.exists():
        MAT_DIR = _p
        break
if MAT_DIR is None:
    raise FileNotFoundError(
        "找不到 MAT 目錄。請確認 ~/Desktop/VOICE/MAT 存在，"
        "或手動修改腳本中的 MAT_DIR 路徑。"
    )

OUTPUT_DIR = PROJECT_DIR / "test_parallel"
OUTPUT_DIR.mkdir(exist_ok=True)

# v1 報告路徑（串行版 threshold=0.25）
V1_REPORT = PROJECT_DIR / "test_mixed" / "test_report.json"

THRESHOLD_PARALLEL = 0.25
PVAD_WINDOW_SEC = 1.0
EMA_ALPHA = 0.6

# ── 音頻檔案 ────────────────────────────────────────
HSUAN_ENROLLMENT = MAT_DIR / "b6dbc0fc-1d57-4647-aa85-54f9bea08743_hsuan_7.wav"
HSUAN_SOURCE     = MAT_DIR / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"
INTERFERER_0911  = MAT_DIR / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"
INTERFERER_FEMH  = MAT_DIR / "ef81890f-791f-45c5-925a-2e7931a8f3c6_FEMH_7.wav"


# ── 工具函式 ─────────────────────────────────────────

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
    """交替說話，每段 2.5 秒，共 5 段。"""
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
    """hsuan 全程說話，中間有 interferer 重疊。"""
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
    """交替說話 + 白噪 SNR 10dB。"""
    mixed_clean, labels = build_scenario_a(hsuan_audio, interferer_audio, seg_sec=seg_sec)
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak
    return mixed_noisy, labels


# ── 串行版管線（v1 重跑，用於 scenario C 對比）────────

def run_serial_pipeline(enrollment_path, input_audio, denoiser, speaker_encoder,
                        threshold=0.25, gain_floor=0.0, pvad_interval=32):
    """
    串行版管線（v1）：GTCRN → pVAD(denoised) → gating。
    用於 scenario C 的 v1 vs v2 對比。
    """
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
    gate = SoftGate(gain_floor=gain_floor, attack_ms=5.0, release_ms=30.0, hop=GTCRN_HOP)

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
        "denoised_audio": denoised_audio[:len(input_audio)],
        "similarities": np.array(similarities),
        "is_targets": np.array(is_targets),
        "n_frames": n_frames,
    }


# ── 視覺化 ───────────────────────────────────────────

def plot_pvad_timeline(result, labels, title, output_path, threshold):
    """繪製 pVAD similarity 時間軸曲線圖。"""
    sims = result["similarities"]
    n_frames = result["n_frames"]
    frame_times = np.arange(n_frames) * GTCRN_HOP / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1) Similarity 曲線
    ax1 = axes[0]
    ax1.plot(frame_times, sims, linewidth=0.8, color="steelblue", label="pVAD similarity")
    ax1.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7,
                label=f"threshold={threshold}")

    colors = {"hsuan": "#2ca02c", "interferer": "#d62728",
              "hsuan_only": "#2ca02c", "hsuan+interferer": "#ff7f0e"}
    for (s, e, spk) in labels:
        ax1.axvspan(s, e, alpha=0.15, color=colors.get(spk, "#999"))
    for i in range(1, len(labels)):
        ax1.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)

    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title(f"{title} — Parallel pVAD (threshold={threshold})")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # 2) Target 判定
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

    # 3) 波形
    ax3 = axes[2]
    denoised_t = np.arange(len(result["denoised_audio"])) / SAMPLE_RATE
    output_t = np.arange(len(result["output_audio"])) / SAMPLE_RATE
    ax3.plot(denoised_t, result["denoised_audio"], alpha=0.4, linewidth=0.3, color="gray",
             label="denoised (GTCRN)")
    ax3.plot(output_t, result["output_audio"], alpha=0.7, linewidth=0.3, color="steelblue",
             label="output (parallel)")
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


def plot_v1_vs_v2_comparison(v1_result, v2_result, labels, output_path):
    """
    場景 C 的 v1（串行）vs v2（並行）對比圖。
    上方：兩版 similarity 疊在一起
    下方：兩版 output 波形
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    n1 = v1_result["n_frames"]
    n2 = v2_result["n_frames"]
    ft1 = np.arange(n1) * GTCRN_HOP / SAMPLE_RATE
    ft2 = np.arange(n2) * GTCRN_HOP / SAMPLE_RATE

    colors = {"hsuan": "#2ca02c", "interferer": "#d62728"}

    # 1) Similarity 對比
    ax1 = axes[0]
    ax1.plot(ft1, v1_result["similarities"], linewidth=0.8, color="coral",
             alpha=0.8, label="v1 serial (pVAD on denoised)")
    ax1.plot(ft2, v2_result["similarities"], linewidth=0.8, color="steelblue",
             alpha=0.8, label="v2 parallel (pVAD on raw)")
    ax1.axhline(y=0.25, color="coral", linestyle=":", linewidth=1, alpha=0.5,
                label="v1 threshold=0.25")
    ax1.axhline(y=0.30, color="steelblue", linestyle="--", linewidth=1, alpha=0.5,
                label="v2 threshold=0.30")
    for (s, e, spk) in labels:
        ax1.axvspan(s, e, alpha=0.12, color=colors.get(spk, "#999"))
    for i in range(1, len(labels)):
        ax1.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Scenario C: v1 (serial) vs v2 (parallel) — pVAD Similarity")
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 2) Target 判定對比
    ax2 = axes[1]
    ax2.fill_between(ft1, v1_result["is_targets"].astype(float),
                     step="mid", alpha=0.3, color="coral", label="v1 is_target")
    ax2.fill_between(ft2, v2_result["is_targets"].astype(float) * 0.8,
                     step="mid", alpha=0.5, color="steelblue", label="v2 is_target (×0.8 for vis)")
    for (s, e, spk) in labels:
        ax2.axvspan(s, e, alpha=0.08, color=colors.get(spk, "#999"))
    for i in range(1, len(labels)):
        ax2.axvline(x=labels[i][0], color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Target Active")
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) Output 波形對比
    ax3 = axes[2]
    t1 = np.arange(len(v1_result["output_audio"])) / SAMPLE_RATE
    t2 = np.arange(len(v2_result["output_audio"])) / SAMPLE_RATE
    ax3.plot(t1, v1_result["output_audio"], alpha=0.5, linewidth=0.3, color="coral",
             label="v1 output (serial)")
    ax3.plot(t2, v2_result["output_audio"], alpha=0.6, linewidth=0.3, color="steelblue",
             label="v2 output (parallel)")
    for (s, e, spk) in labels:
        ax3.axvspan(s, e, alpha=0.08, color=colors.get(spk, "#999"))
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  v1 vs v2 chart saved: {output_path}")


def compute_metrics(result, labels, threshold):
    """計算各段的 pVAD 指標。"""
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


# ── 主流程 ───────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 60)
    print("並行管線測試 v3 (CAM++ + EMA)")
    print(f"  threshold: {THRESHOLD_PARALLEL}")
    print(f"  pVAD window: {PVAD_WINDOW_SEC}s, EMA alpha: {EMA_ALPHA}")
    print(f"  augmented enrollment: False (clean)")
    print("=" * 60)

    # ── 載入模型 ──────────────────────────────────────
    print("\n載入模型...")
    campp_path = MODELS_DIR / "campplus" / "campplus.onnx"
    if not campp_path.exists():
        campp_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
        print("  [WARN] CAM++ not found, using WeSpeaker fallback")
    assert campp_path.exists(), f"找不到 speaker encoder: {campp_path}"
    speaker_encoder = SpeakerEncoder(str(campp_path))
    denoiser = GTCRNDenoiser()
    print(f"  model: {campp_path.name}, embed_dim={speaker_encoder.embed_dim}")
    print("  done")

    # ── 載入語音素材 ──────────────────────────────────
    print("\n載入語音素材...")
    print(f"  MAT 目錄: {MAT_DIR}")
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

    # 儲存 scenario C 的結果供 v1 vs v2 對比
    sc_c_result_v2 = None
    sc_c_labels = None
    sc_c_mixed = None

    for sc_id, sc_title, build_fn in scenarios:
        print(f"\n{'=' * 50}")
        print(f"場景 {sc_title}")
        print("=" * 50)

        mixed, labels = build_fn()
        write_audio(str(OUTPUT_DIR / f"{sc_id}_mixed.wav"), mixed)
        print(f"  mixed: {len(mixed)/SAMPLE_RATE:.1f}s, segments: {len(labels)}")

        t0 = time.time()
        result = run_parallel_pipeline_from_audio(
            enrollment_path=HSUAN_ENROLLMENT,
            input_audio=mixed,
            denoiser=denoiser,
            speaker_encoder=speaker_encoder,
            threshold=THRESHOLD_PARALLEL,
            pvad_interval=32,
            pvad_window_sec=PVAD_WINDOW_SEC,
            ema_alpha=EMA_ALPHA,
            use_augmented_enrollment=True,
            denoise_enrollment=False,
        )
        elapsed = time.time() - t0
        print(f"  pipeline 完成: {elapsed:.1f}s")

        write_audio(str(OUTPUT_DIR / f"{sc_id}_output.wav"), result["output_audio"])
        write_audio(str(OUTPUT_DIR / f"{sc_id}_denoised.wav"), result["denoised_audio"])

        plot_pvad_timeline(result, labels, f"Scenario {sc_title}",
                           OUTPUT_DIR / f"{sc_id}_pvad.png",
                           threshold=THRESHOLD_PARALLEL)

        metrics = compute_metrics(result, labels, threshold=THRESHOLD_PARALLEL)
        all_metrics[sc_id] = metrics

        # 印出各段指標
        for seg_name, m in metrics.items():
            is_hsuan = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            marker = "[O] TARGET" if is_hsuan else "[X] INTERF"
            print(f"  {marker} [{seg_name}]: "
                  f"sim={m['mean_similarity']:.3f}±{m['std_similarity']:.3f}, "
                  f"target_ratio={m['target_ratio']:.1%}")

        # 儲存 scenario C
        if sc_id == "scenario_c":
            sc_c_result_v2 = result
            sc_c_labels = labels
            sc_c_mixed = mixed

    # ── 場景 C: v1 vs v2 對比 ────────────────────────
    print(f"\n{'=' * 50}")
    print("場景 C: v1 (serial) vs v2 (parallel) 對比")
    print("=" * 50)

    if sc_c_mixed is not None:
        print("  重跑 v1（串行版，threshold=0.25）...")
        t0 = time.time()
        sc_c_result_v1 = run_serial_pipeline(
            enrollment_path=HSUAN_ENROLLMENT,
            input_audio=sc_c_mixed,
            denoiser=denoiser,
            speaker_encoder=speaker_encoder,
            threshold=0.25,
            pvad_interval=32,
        )
        print(f"  v1 完成: {time.time()-t0:.1f}s")

        write_audio(str(OUTPUT_DIR / "scenario_c_v1_output.wav"), sc_c_result_v1["output_audio"])

        # 對比圖
        plot_v1_vs_v2_comparison(
            sc_c_result_v1, sc_c_result_v2, sc_c_labels,
            OUTPUT_DIR / "scenario_c_v1_vs_v2.png"
        )

        # 對比指標
        v1_metrics_c = compute_metrics(sc_c_result_v1, sc_c_labels, threshold=0.25)
        v2_metrics_c = all_metrics.get("scenario_c", {})

        print("\n  v1 (serial, threshold=0.25) vs v2 (parallel, threshold=0.30):")
        all_segs = sorted(set(list(v1_metrics_c.keys()) + list(v2_metrics_c.keys())))
        for seg in all_segs:
            is_hsuan = "hsuan" in seg and "interferer" not in seg and "+" not in seg
            marker = "TARGET" if is_hsuan else "INTERF"
            v1m = v1_metrics_c.get(seg, {})
            v2m = v2_metrics_c.get(seg, {})
            v1_sim = v1m.get("mean_similarity", float("nan"))
            v2_sim = v2m.get("mean_similarity", float("nan"))
            v1_tr = v1m.get("target_ratio", float("nan"))
            v2_tr = v2m.get("target_ratio", float("nan"))
            delta_tr = v2_tr - v1_tr if not (np.isnan(v1_tr) or np.isnan(v2_tr)) else float("nan")
            print(f"    [{marker}] {seg}")
            print(f"      sim: {v1_sim:.3f} → {v2_sim:.3f}  |  "
                  f"target_ratio: {v1_tr:.1%} → {v2_tr:.1%} ({delta_tr:+.1%})")

        # 儲存 v1 vs v2 對比 JSON
        comparison_c = {}
        for seg in all_segs:
            comparison_c[seg] = {
                "v1_serial_threshold_0.25": v1_metrics_c.get(seg, {}),
                "v2_parallel_threshold_0.30": v2_metrics_c.get(seg, {}),
            }
        comp_path = OUTPUT_DIR / "scenario_c_v1_vs_v2.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison_c, f, indent=2, ensure_ascii=False)
        print(f"\n  v1 vs v2 comparison saved: {comp_path}")

    # ── 儲存完整報告 ──────────────────────────────────
    report = {
        "config": {
            "pipeline": "parallel (pVAD on raw audio)",
            "threshold": THRESHOLD_PARALLEL,
            "pvad_window_sec": PVAD_WINDOW_SEC,
            "augmented_enrollment": False,
            "denoise_enrollment": False,
            "gain_floor": 0.0,
            "attack_ms": 5.0,
            "release_ms": 30.0,
            "pvad_interval": 32,
        },
        "metrics": all_metrics,
    }
    report_path = OUTPUT_DIR / "test_report_parallel.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n報告已存: {report_path}")

    # ── 與 v1 報告全面比較 ────────────────────────────
    if V1_REPORT.exists():
        print(f"\n{'=' * 60}")
        print("全場景 v1 (serial) vs v2 (parallel) 總覽")
        print("=" * 60)
        with open(V1_REPORT, "r") as f:
            v1_all = json.load(f)

        for sc in ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]:
            v1s = v1_all.get(sc, {})
            v2s = all_metrics.get(sc, {})
            if not v1s and not v2s:
                continue
            print(f"\n  {sc}:")
            all_seg_keys = sorted(set(list(v1s.keys()) + list(v2s.keys())))
            for seg in all_seg_keys:
                is_hsuan = "hsuan" in seg and "interferer" not in seg and "+" not in seg
                marker = "TARGET" if is_hsuan else "INTERF"
                v1m = v1s.get(seg, {})
                v2m = v2s.get(seg, {})
                v1_tr = v1m.get("target_ratio", float("nan"))
                v2_tr = v2m.get("target_ratio", float("nan"))
                delta = v2_tr - v1_tr if not (np.isnan(v1_tr) or np.isnan(v2_tr)) else float("nan")
                # 期望：hsuan 段 target_ratio 高，interferer 段 target_ratio 低
                if is_hsuan:
                    quality = "OK" if not np.isnan(v2_tr) and v2_tr > 0.5 else "?"
                else:
                    quality = "OK" if not np.isnan(v2_tr) and v2_tr < 0.3 else "?"
                print(f"    {quality} [{marker}] {seg}: "
                      f"v1={v1_tr:.1%} → v2={v2_tr:.1%} ({delta:+.1%})")

    print(f"\n所有輸出存在: {OUTPUT_DIR}")
    print("完成！")


if __name__ == "__main__":
    main()
