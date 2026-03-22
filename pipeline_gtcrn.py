#!/usr/bin/env python3
"""
離線版管線（pVAD + GTCRN SE）
==============================
B 路線升級版：用 GTCRN 取代 DTLN，搭配 WeSpeaker ResNet34 做 pVAD。

流程：
1. 載入 enrollment 音頻 → 用 WeSpeaker ResNet34 提取 d-vector
2. 載入混合音頻
3. GTCRN 離線降噪（逐幀 STFT → 模型推論 → iSTFT → OLA）
4. CachedPVAD：每 N 幀提取一次 embedding 做 cosine similarity
5. Soft gating 輸出

用法:
    python pipeline_gtcrn.py --enrollment enroll.wav --input mixed.wav --output output.wav
"""

import argparse
import time
import numpy as np
from pathlib import Path

from utils.audio import SAMPLE_RATE, read_audio, write_audio, frame_signal, overlap_add
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


def run_pipeline(enrollment_path: str, input_path: str, output_path: str,
                 threshold: float = 0.25,
                 gain_floor: float = 0.05,
                 attack_ms: float = 5.0,
                 release_ms: float = 50.0,
                 pvad_interval: int = 32):
    """
    GTCRN + CachedPVAD 離線管線。
    """
    print("\n" + "=" * 60)
    print("pVAD + GTCRN SE 離線管線")
    print("=" * 60)

    # ── 1. 載入模型 ──────────────────────────────────
    print("\n[1/5] 載入模型...")
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    assert ecapa_path.exists(), f"找不到 speaker encoder ONNX: {ecapa_path}"

    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    denoiser = GTCRNDenoiser()
    gate = SoftGate(
        gain_floor=gain_floor,
        attack_ms=attack_ms,
        release_ms=release_ms,
        hop=GTCRN_HOP,
    )
    print("  模型載入完成")

    # ── 2. 提取 enrollment d-vector ──────────────────
    print("\n[2/5] 提取 enrollment d-vector...")
    enrollment_audio = read_audio(enrollment_path)
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)
    print(f"  enrollment 長度: {len(enrollment_audio) / SAMPLE_RATE:.2f}s")
    print(f"  d-vector 維度: {enrollment_dvector.shape}")

    # ── 3. 載入混合音頻 ──────────────────────────────
    print("\n[3/5] 載入混合音頻...")
    input_audio = read_audio(input_path)
    print(f"  混合音頻長度: {len(input_audio) / SAMPLE_RATE:.2f}s")

    # ── 4. GTCRN 離線降噪 ────────────────────────────
    print("\n[4/5] GTCRN 降噪 + pVAD gating...")
    t_start = time.time()

    # 先做整段降噪
    denoised_audio = denoiser.enhance(input_audio)

    t_denoise = time.time() - t_start
    print(f"  GTCRN 降噪完成: {t_denoise:.2f}s")

    # 對降噪後的音頻做 pVAD + gating（用 GTCRN 的 hop 切幀）
    pvad = CachedPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        threshold=threshold,
    )

    # 用 GTCRN 的 hop 切幀
    frames = frame_signal(denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = frames.shape[0]
    print(f"  pVAD 總幀數: {n_frames}, 每 {pvad_interval} 幀提取一次 embedding")

    enhanced_frames = np.empty_like(frames)
    similarities = []

    for i in range(n_frames):
        # pVAD 用 hop-size 的 samples
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(denoised_audio))
        frame_samples = denoised_audio[start:end]

        is_target, sim = pvad.process_frame(frame_samples)
        similarities.append(sim)

        enhanced_frames[i] = gate.process(frames[i], is_target, confidence=sim)

        if (i + 1) % 500 == 0 or i == n_frames - 1:
            pct = (i + 1) / n_frames * 100
            avg_sim = np.mean(similarities[-100:]) if similarities else 0
            print(f"  [{pct:5.1f}%] frame {i+1}/{n_frames}, "
                  f"avg_sim={avg_sim:.3f}, target={is_target}")

    elapsed = time.time() - t_start
    rtf = elapsed / (len(input_audio) / SAMPLE_RATE)
    print(f"\n  總處理時間: {elapsed:.2f}s, RTF: {rtf:.3f}")

    # ── 5. 合成輸出 ──────────────────────────────────
    print("\n[5/5] 合成輸出音頻...")
    output_audio = overlap_add(enhanced_frames, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)

    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val
        print(f"  正規化: peak {max_val:.3f} -> 0.99")

    # 裁剪到原始長度
    output_audio = output_audio[:len(input_audio)]
    write_audio(output_path, output_audio)
    print(f"  輸出: {output_path}")

    # ── 統計 ──────────────────────────────────────────
    sims = np.array(similarities)
    print(f"\npVAD 統計:")
    print(f"  similarity 平均: {sims.mean():.3f}")
    print(f"  similarity 標準差: {sims.std():.3f}")
    print(f"  目標活躍幀比例: {(sims > threshold).mean():.1%}")
    print(f"  embedding 提取次數: {n_frames // pvad_interval} (vs 原本每幀 {n_frames} 次)")


def main():
    parser = argparse.ArgumentParser(description="pVAD + GTCRN SE 離線管線")
    parser.add_argument("--enrollment", "-e", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="output_gtcrn.wav")
    parser.add_argument("--threshold", "-t", type=float, default=0.25)
    parser.add_argument("--gain-floor", type=float, default=0.05)
    parser.add_argument("--attack-ms", type=float, default=5.0)
    parser.add_argument("--release-ms", type=float, default=50.0)
    parser.add_argument("--pvad-interval", type=int, default=32,
                        help="pVAD embedding 提取間隔（幀數，預設 32 = 0.5s）")
    args = parser.parse_args()

    run_pipeline(
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
