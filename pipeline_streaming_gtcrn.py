#!/usr/bin/env python3
"""
串流版管線（GTCRN + CachedPVAD，模擬即時處理）
================================================
模擬即時串流：
- 每次讀入 GTCRN_HOP (256 samples = 16ms) 的新資料
- GTCRN 逐幀降噪（維護 conv/tra/inter caches）
- CachedPVAD 每 N 幀提取一次 embedding
- Soft gating 輸出
- 測量每幀延遲和 RTF

用法:
    python pipeline_streaming_gtcrn.py --enrollment enroll.wav --input mixed.wav --output output.wav
"""

import argparse
import time
import numpy as np
from pathlib import Path

from utils.audio import SAMPLE_RATE, read_audio, write_audio
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD
from utils.gtcrn_denoiser import StreamingGTCRNDenoiser, GTCRN_HOP

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


def run_streaming_pipeline(
    enrollment_path: str,
    input_path: str,
    output_path: str,
    threshold: float = 0.25,
    gain_floor: float = 0.05,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    pvad_interval: int = 32,
):
    """
    GTCRN + CachedPVAD 串流管線。
    """
    print("\n" + "=" * 60)
    print("pVAD + GTCRN SE 串流管線（模擬即時）")
    print("=" * 60)

    # ── 載入模型 ──────────────────────────────────────
    print("\n[1/4] 載入模型...")
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    denoiser = StreamingGTCRNDenoiser()
    gate = SoftGate(
        gain_floor=gain_floor,
        attack_ms=attack_ms,
        release_ms=release_ms,
        hop=GTCRN_HOP,
    )
    print("  模型載入完成")

    # ── 提取 enrollment d-vector ──────────────────────
    print("\n[2/4] 提取 enrollment d-vector...")
    enrollment_audio = read_audio(enrollment_path)
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)
    print(f"  d-vector shape: {enrollment_dvector.shape}")

    # ── 建立 CachedPVAD ──────────────────────────────
    pvad = CachedPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        threshold=threshold,
    )

    # ── 載入輸入音頻 ──────────────────────────────────
    print("\n[3/4] 載入混合音頻...")
    input_audio = read_audio(input_path)
    total_samples = len(input_audio)
    total_shifts = total_samples // GTCRN_HOP
    duration_s = total_samples / SAMPLE_RATE
    print(f"  長度: {duration_s:.2f}s, 總 shifts: {total_shifts}")

    # ── 串流處理 ──────────────────────────────────────
    print(f"\n[4/4] 串流處理中（pVAD 每 {pvad_interval} 幀提取一次 embedding）...")

    output_samples_list = []
    frame_times = []
    frame_times_denoise = []
    frame_times_pvad = []
    similarities = []

    padded_len = total_shifts * GTCRN_HOP
    padded_audio = np.zeros(padded_len, dtype=np.float32)
    padded_audio[:min(total_samples, padded_len)] = input_audio[:padded_len]

    for i in range(total_shifts):
        t0 = time.perf_counter()

        start = i * GTCRN_HOP
        new_samples = padded_audio[start:start + GTCRN_HOP]

        # GTCRN 降噪
        t_d0 = time.perf_counter()
        denoised_shift = denoiser.process_shift(new_samples)
        t_d1 = time.perf_counter()
        frame_times_denoise.append(t_d1 - t_d0)

        # CachedPVAD
        t_p0 = time.perf_counter()
        is_target, sim = pvad.process_frame(denoised_shift)
        t_p1 = time.perf_counter()
        frame_times_pvad.append(t_p1 - t_p0)
        similarities.append(sim)

        # Soft gating
        gated_shift = gate.process(denoised_shift, is_target, confidence=sim)
        output_samples_list.append(gated_shift)

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        # 進度（每秒報告一次）
        frames_per_sec = SAMPLE_RATE // GTCRN_HOP
        if (i + 1) % frames_per_sec == 0:
            sec = (i + 1) * GTCRN_HOP / SAMPLE_RATE
            avg_ft = np.mean(frame_times[-frames_per_sec:]) * 1000
            avg_dn = np.mean(frame_times_denoise[-frames_per_sec:]) * 1000
            avg_pv = np.mean(frame_times_pvad[-frames_per_sec:]) * 1000
            print(f"  [{sec:6.1f}s / {duration_s:.1f}s] "
                  f"total={avg_ft:.2f}ms (denoise={avg_dn:.2f}ms + pvad={avg_pv:.2f}ms), "
                  f"sim={sim:.3f}, target={is_target}")

    # ── 合成輸出 ──────────────────────────────────────
    output_audio = np.concatenate(output_samples_list)[:total_samples]

    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val

    write_audio(output_path, output_audio)
    print(f"\n  輸出: {output_path}")

    # ── 延遲統計 ──────────────────────────────────────
    frame_times_ms = np.array(frame_times) * 1000
    denoise_ms = np.array(frame_times_denoise) * 1000
    pvad_ms = np.array(frame_times_pvad) * 1000
    shift_duration_ms = GTCRN_HOP / SAMPLE_RATE * 1000  # 16ms

    total_rtf = frame_times_ms.mean() / shift_duration_ms

    print(f"\n串流延遲統計:")
    print(f"  幀持續時間（即時約束）: {shift_duration_ms:.1f} ms")
    print(f"  --- 總計 ---")
    print(f"  平均每幀推論: {frame_times_ms.mean():.2f} ms")
    print(f"  P95: {np.percentile(frame_times_ms, 95):.2f} ms")
    print(f"  P99: {np.percentile(frame_times_ms, 99):.2f} ms")
    print(f"  RTF: {total_rtf:.3f}")
    print(f"  --- GTCRN 降噪 ---")
    print(f"  平均: {denoise_ms.mean():.2f} ms")
    print(f"  P95: {np.percentile(denoise_ms, 95):.2f} ms")
    print(f"  --- CachedPVAD ---")
    print(f"  平均: {pvad_ms.mean():.2f} ms (含非提取幀)")
    print(f"  P95: {np.percentile(pvad_ms, 95):.2f} ms")

    if total_rtf < 1.0:
        print(f"  可即時處理 (RTF < 1)")
    else:
        print(f"  無法即時 (RTF >= 1)，需優化")

    # pVAD 統計
    sims = np.array(similarities)
    print(f"\npVAD 統計:")
    print(f"  similarity 平均: {sims.mean():.3f}")
    print(f"  similarity 標準差: {sims.std():.3f}")
    print(f"  目標活躍比例: {(sims > threshold).mean():.1%}")

    return {
        'rtf': total_rtf,
        'avg_frame_ms': frame_times_ms.mean(),
        'avg_denoise_ms': denoise_ms.mean(),
        'avg_pvad_ms': pvad_ms.mean(),
        'p95_frame_ms': np.percentile(frame_times_ms, 95),
        'sim_mean': sims.mean(),
        'sim_std': sims.std(),
    }


def main():
    parser = argparse.ArgumentParser(description="pVAD + GTCRN SE 串流管線")
    parser.add_argument("--enrollment", "-e", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="output_streaming_gtcrn.wav")
    parser.add_argument("--threshold", "-t", type=float, default=0.25)
    parser.add_argument("--gain-floor", type=float, default=0.05)
    parser.add_argument("--attack-ms", type=float, default=5.0)
    parser.add_argument("--release-ms", type=float, default=50.0)
    parser.add_argument("--pvad-interval", type=int, default=32,
                        help="pVAD embedding 提取間隔（預設 32 = 0.5s at 16ms hop）")
    args = parser.parse_args()

    run_streaming_pipeline(
        enrollment_path=args.enrollment,
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        gain_floor=args.gain_floor,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
        pvad_interval=args.pvad_interval,
    )


if __name__ == "__main__":
    main()
