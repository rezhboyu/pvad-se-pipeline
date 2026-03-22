#!/usr/bin/env python3
"""
串流版管線（模擬即時處理）
===========================
模擬即時串流處理：
- 每次只讀入一幀（BLOCK_SHIFT = 128 samples = 8ms）
- 維護 DTLN 的 LSTM hidden states
- 用 ONNX Runtime 推論
- 測量每幀推論時間

與離線版的差異：
- 使用 overlap-add 的串流模式（逐幀推入、逐幀推出）
- 維護所有模組的內部狀態
- 測量即時性指標（RTF、每幀延遲）

用法:
    python pipeline_streaming.py --enrollment enroll.wav --input mixed.wav --output output.wav
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import deque

from utils.audio import (
    SAMPLE_RATE, BLOCK_LEN, BLOCK_SHIFT, N_FREQ_BINS,
    read_audio, write_audio,
    stft_frame, istft_frame, magnitude_phase, reconstruct_complex,
)
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, SimplePVAD

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


# ═══════════════════════════════════════════════════════
# 串流 DTLN 降噪器
# ═══════════════════════════════════════════════════════
class StreamingDTLNDenoiser:
    """
    串流版 DTLN 降噪器。

    與離線版的差異：
    - 維護一個長度為 BLOCK_LEN 的輸入緩衝區
    - 每次推入 BLOCK_SHIFT 個新 samples
    - 持續追蹤 LSTM hidden states
    - 輸出緩衝區做 overlap-add
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(MODELS_DIR / "dtln")

        model_dir = Path(model_dir)
        model_1_path = model_dir / "dtln_1.onnx"
        model_2_path = model_dir / "dtln_2.onnx"

        assert model_1_path.exists(), f"找不到 DTLN model 1: {model_1_path}"
        assert model_2_path.exists(), f"找不到 DTLN model 2: {model_2_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1  # 串流版用單執行緒降低延遲

        self.session_1 = ort.InferenceSession(
            str(model_1_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.session_2 = ort.InferenceSession(
            str(model_2_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.in1_names = [i.name for i in self.session_1.get_inputs()]
        self.out1_names = [o.name for o in self.session_1.get_outputs()]
        self.in2_names = [i.name for i in self.session_2.get_inputs()]
        self.out2_names = [o.name for o in self.session_2.get_outputs()]

        # 輸入緩衝區：累積到 BLOCK_LEN 才做一次 STFT
        self.input_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        # 輸出緩衝區：overlap-add
        self.output_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)

        # LSTM hidden states
        self.states_1 = None
        self.states_2 = None

    def reset(self):
        """重設所有內部狀態。"""
        self.input_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        self.output_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        self.states_1 = None
        self.states_2 = None

    def process_shift(self, new_samples: np.ndarray):
        """
        推入 BLOCK_SHIFT 個新 samples，回傳 BLOCK_SHIFT 個增強 samples。

        這模擬了即時串流：每 8ms 推入新資料，立即得到輸出。

        Parameters
        ----------
        new_samples : (BLOCK_SHIFT,) 新的時域 samples

        Returns
        -------
        output_samples : (BLOCK_SHIFT,) 增強後的 samples
        """
        assert len(new_samples) == BLOCK_SHIFT, \
            f"期望 {BLOCK_SHIFT} 個 samples，得到 {len(new_samples)}"

        # 移動輸入緩衝區，加入新 samples
        self.input_buffer = np.roll(self.input_buffer, -BLOCK_SHIFT)
        self.input_buffer[-BLOCK_SHIFT:] = new_samples

        # 對整個緩衝區做 STFT
        spectrum = stft_frame(self.input_buffer)
        mag, phase = magnitude_phase(spectrum)

        # ── Stage 1 ──
        mag_input = mag[np.newaxis, np.newaxis, :]

        if len(self.in1_names) == 1:
            feed_1 = {self.in1_names[0]: mag_input}
        else:
            feed_1 = {self.in1_names[0]: mag_input}
            if self.states_1 is not None:
                for i, name in enumerate(self.in1_names[1:]):
                    if i == 0:
                        feed_1[name] = self.states_1["h"]
                    else:
                        feed_1[name] = self.states_1["c"]

        out_1 = self.session_1.run(self.out1_names, feed_1)
        mask = out_1[0].squeeze()

        # 更新 states
        if len(out_1) > 1:
            if self.states_1 is None:
                self.states_1 = {"h": None, "c": None}
            self.states_1["h"] = out_1[1] if len(out_1) > 1 else self.states_1["h"]
            self.states_1["c"] = out_1[2] if len(out_1) > 2 else self.states_1["c"]

        # 應用 mask
        enhanced_mag = mag * mask
        enhanced_spectrum = reconstruct_complex(enhanced_mag, phase)
        enhanced_frame = istft_frame(enhanced_spectrum)

        # ── Stage 2 ──
        frame_input = enhanced_frame[np.newaxis, np.newaxis, :]

        if len(self.in2_names) == 1:
            feed_2 = {self.in2_names[0]: frame_input}
        else:
            feed_2 = {self.in2_names[0]: frame_input}
            if self.states_2 is not None:
                for i, name in enumerate(self.in2_names[1:]):
                    if i == 0:
                        feed_2[name] = self.states_2["h"]
                    else:
                        feed_2[name] = self.states_2["c"]

        out_2 = self.session_2.run(self.out2_names, feed_2)
        enhanced_frame_2 = out_2[0].squeeze()

        if len(out_2) > 1:
            if self.states_2 is None:
                self.states_2 = {"h": None, "c": None}
            self.states_2["h"] = out_2[1] if len(out_2) > 1 else self.states_2["h"]
            self.states_2["c"] = out_2[2] if len(out_2) > 2 else self.states_2["c"]

        # Overlap-add 到輸出緩衝區
        # 先取出最早的 BLOCK_SHIFT 個 samples 作為輸出
        output_samples = self.output_buffer[:BLOCK_SHIFT].copy()

        # 移動緩衝區
        self.output_buffer = np.roll(self.output_buffer, -BLOCK_SHIFT)
        self.output_buffer[-BLOCK_SHIFT:] = 0.0

        # 加入新的增強幀
        self.output_buffer += enhanced_frame_2[:BLOCK_LEN]

        return output_samples.astype(np.float32)


# ═══════════════════════════════════════════════════════
# 串流管線
# ═══════════════════════════════════════════════════════
def run_streaming_pipeline(
    enrollment_path: str,
    input_path: str,
    output_path: str,
    threshold: float = 0.25,
    gain_floor: float = 0.05,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
):
    """
    串流版管線：模擬即時處理並測量延遲。
    """
    print("\n" + "=" * 60)
    print("pVAD + SE 串流管線（模擬即時）")
    print("=" * 60)

    # ── 載入模型 ──────────────────────────────────────
    print("\n[1/4] 載入模型...")
    # 優先使用 WeSpeaker ResNet34-LM（官方驗證的 ONNX，鑑別力更佳）
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    denoiser = StreamingDTLNDenoiser()
    gate = SoftGate(
        gain_floor=gain_floor,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    print("  ✅ 模型載入完成")

    # ── 提取 enrollment d-vector ──────────────────────
    print("\n[2/4] 提取 enrollment d-vector...")
    enrollment_audio = read_audio(enrollment_path)
    enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)
    print(f"  d-vector shape: {enrollment_dvector.shape}")

    # ── 建立 pVAD ─────────────────────────────────────
    pvad = SimplePVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        threshold=threshold,
    )

    # ── 載入輸入音頻 ──────────────────────────────────
    print("\n[3/4] 載入混合音頻...")
    input_audio = read_audio(input_path)
    total_samples = len(input_audio)
    total_shifts = total_samples // BLOCK_SHIFT
    duration_s = total_samples / SAMPLE_RATE
    print(f"  長度: {duration_s:.2f}s, 總 shifts: {total_shifts}")

    # ── 串流處理 ──────────────────────────────────────
    print("\n[4/4] 串流處理中...")

    output_samples_list = []
    frame_times = []
    similarities = []

    # 在輸入末尾補零，確保整除 BLOCK_SHIFT
    padded_len = total_shifts * BLOCK_SHIFT
    padded_audio = np.zeros(padded_len, dtype=np.float32)
    padded_audio[:total_samples] = input_audio[:padded_len]

    for i in range(total_shifts):
        t0 = time.perf_counter()

        # 取出這一 shift 的 samples
        start = i * BLOCK_SHIFT
        new_samples = padded_audio[start:start + BLOCK_SHIFT]

        # DTLN 降噪
        denoised_shift = denoiser.process_shift(new_samples)

        # pVAD
        is_target, sim = pvad.process_frame(denoised_shift)
        similarities.append(sim)

        # Soft gating
        gated_shift = gate.process(denoised_shift, is_target, confidence=sim)
        output_samples_list.append(gated_shift)

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        # 進度報告（每秒報告一次）
        if (i + 1) % (SAMPLE_RATE // BLOCK_SHIFT) == 0:
            sec = (i + 1) * BLOCK_SHIFT / SAMPLE_RATE
            avg_ft = np.mean(frame_times[-100:]) * 1000
            print(f"  [{sec:6.1f}s / {duration_s:.1f}s] "
                  f"avg_frame_time={avg_ft:.2f}ms, "
                  f"sim={sim:.3f}, target={is_target}")

    # ── 合成輸出 ──────────────────────────────────────
    output_audio = np.concatenate(output_samples_list)[:total_samples]

    # 正規化
    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val

    write_audio(output_path, output_audio)
    print(f"\n  ✅ 輸出: {output_path}")

    # ── 延遲統計 ──────────────────────────────────────
    frame_times_ms = np.array(frame_times) * 1000
    shift_duration_ms = BLOCK_SHIFT / SAMPLE_RATE * 1000  # 8ms

    print(f"\n📊 串流延遲統計:")
    print(f"  幀持續時間（即時約束）: {shift_duration_ms:.1f} ms")
    print(f"  平均每幀推論時間: {frame_times_ms.mean():.2f} ms")
    print(f"  中位數: {np.median(frame_times_ms):.2f} ms")
    print(f"  P95: {np.percentile(frame_times_ms, 95):.2f} ms")
    print(f"  P99: {np.percentile(frame_times_ms, 99):.2f} ms")
    print(f"  最大值: {frame_times_ms.max():.2f} ms")
    print(f"  RTF: {frame_times_ms.mean() / shift_duration_ms:.3f}")

    if frame_times_ms.mean() < shift_duration_ms:
        print(f"  ✅ 平均推論時間 < 幀持續時間 → 可即時處理")
    else:
        print(f"  ⚠ 平均推論時間 > 幀持續時間 → 無法即時，需要優化")

    # pVAD 統計
    sims = np.array(similarities)
    print(f"\n📊 pVAD 統計:")
    print(f"  similarity 平均: {sims.mean():.3f}")
    print(f"  similarity 標準差: {sims.std():.3f}")
    print(f"  目標活躍比例: {(sims > threshold).mean():.1%}")
    print()


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="pVAD + SE 串流管線（模擬即時）",
    )
    parser.add_argument("--enrollment", "-e", required=True,
                        help="enrollment 音頻路徑")
    parser.add_argument("--input", "-i", required=True,
                        help="混合音頻路徑")
    parser.add_argument("--output", "-o", default="output_streaming.wav",
                        help="輸出音頻路徑")
    parser.add_argument("--threshold", "-t", type=float, default=0.25,
                        help="pVAD 閾值")
    parser.add_argument("--gain-floor", type=float, default=0.05,
                        help="最小增益")
    parser.add_argument("--attack-ms", type=float, default=5.0,
                        help="attack 時間 ms")
    parser.add_argument("--release-ms", type=float, default=50.0,
                        help="release 時間 ms")

    args = parser.parse_args()

    run_streaming_pipeline(
        enrollment_path=args.enrollment,
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        gain_floor=args.gain_floor,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
    )


if __name__ == "__main__":
    main()
