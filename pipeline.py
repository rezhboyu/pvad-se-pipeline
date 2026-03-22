#!/usr/bin/env python3
"""
離線版管線（pVAD + SE）
========================
完整的離線處理流程：

1. 載入 enrollment 音頻 → 用 ECAPA-TDNN 提取 d-vector
2. 載入混合音頻
3. 逐幀處理：
   a. DTLN 降噪（Stage 1 + Stage 2）
   b. 簡易 pVAD：cosine similarity 判定目標說話者
   c. Soft gating with gain floor
4. Overlap-add 合成輸出音頻

用法:
    python pipeline.py --enrollment enroll.wav --input mixed.wav --output output.wav
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

from utils.audio import (
    SAMPLE_RATE, BLOCK_LEN, BLOCK_SHIFT, N_FREQ_BINS,
    read_audio, write_audio,
    stft_frame, istft_frame, magnitude_phase, reconstruct_complex,
    frame_signal, overlap_add,
)
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, SimplePVAD

# ── 路徑設定 ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


# ═══════════════════════════════════════════════════════
# DTLN 降噪器（ONNX 推論）
# ═══════════════════════════════════════════════════════
class DTLNDenoiser:
    """
    DTLN 降噪器，使用兩個 ONNX 模型：
    - model_1: 幅度頻譜域，輸出 mask
    - model_2: 時域特徵，輸出增強訊號

    離線版：一次處理整段音頻，不維護 LSTM hidden states。
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(MODELS_DIR / "dtln")

        model_dir = Path(model_dir)
        model_1_path = model_dir / "dtln_1.onnx"
        model_2_path = model_dir / "dtln_2.onnx"

        assert model_1_path.exists(), f"找不到 DTLN model 1: {model_1_path}"
        assert model_2_path.exists(), f"找不到 DTLN model 2: {model_2_path}"

        # 建立 ONNX Runtime sessions
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        self.session_1 = ort.InferenceSession(
            str(model_1_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.session_2 = ort.InferenceSession(
            str(model_2_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        # 取得輸入 / 輸出名稱
        self.in1_names = [i.name for i in self.session_1.get_inputs()]
        self.out1_names = [o.name for o in self.session_1.get_outputs()]
        self.in2_names = [i.name for i in self.session_2.get_inputs()]
        self.out2_names = [o.name for o in self.session_2.get_outputs()]

        print(f"[DTLN] Model 1 輸入: {self.in1_names}, 輸出: {self.out1_names}")
        print(f"[DTLN] Model 2 輸入: {self.in2_names}, 輸出: {self.out2_names}")

    def _init_states(self):
        """
        初始化 LSTM hidden states。
        DTLN ONNX 模型的 hidden state 形狀: (1, 2, 128, 2)
        其中最後一維的 2 代表 h 和 c 合併在一起。
        """
        # 形狀: (1, n_layers=2, hidden_size=128, h_and_c=2)
        states_1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        states_2 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        return states_1, states_2

    def process_frame(self, frame: np.ndarray,
                      states_1: dict = None,
                      states_2: dict = None):
        """
        處理單一幀。

        Parameters
        ----------
        frame : (BLOCK_LEN,) 時域輸入
        states_1, states_2 : LSTM hidden states（串流用）

        Returns
        -------
        enhanced_frame : (BLOCK_LEN,) 增強後的時域訊號
        states_1, states_2 : 更新的 hidden states
        """
        # ── Stage 1: 幅度域處理 ──
        spectrum = stft_frame(frame)
        mag, phase = magnitude_phase(spectrum)

        # 準備輸入：(1, 1, 257)
        mag_input = mag[np.newaxis, np.newaxis, :]

        # Hidden state 格式: 單一 tensor (1, 2, 128, 2)
        if states_1 is None:
            states_1, _ = self._init_states()

        if len(self.in1_names) == 1:
            feed_1 = {self.in1_names[0]: mag_input}
        else:
            feed_1 = {
                self.in1_names[0]: mag_input,
                self.in1_names[1]: states_1,
            }

        out_1 = self.session_1.run(self.out1_names, feed_1)
        mask = out_1[0]  # (1, 1, 257)

        # 更新 states_1
        if len(out_1) > 1:
            states_1 = out_1[1]

        # 應用 mask
        mask = mask.squeeze()  # (257,)
        enhanced_mag = mag * mask
        enhanced_spectrum = reconstruct_complex(enhanced_mag, phase)

        # ── Stage 2: 時域精煉 ──
        # ISTFT 得到初步增強訊號
        enhanced_frame_1 = istft_frame(enhanced_spectrum)

        # 準備 Stage 2 輸入
        frame_input = enhanced_frame_1[np.newaxis, np.newaxis, :]

        if states_2 is None:
            _, states_2 = self._init_states()

        if len(self.in2_names) == 1:
            feed_2 = {self.in2_names[0]: frame_input}
        else:
            feed_2 = {
                self.in2_names[0]: frame_input,
                self.in2_names[1]: states_2,
            }

        out_2 = self.session_2.run(self.out2_names, feed_2)
        enhanced_frame_2 = out_2[0].squeeze()  # (BLOCK_LEN,)

        if len(out_2) > 1:
            states_2 = out_2[1]

        return enhanced_frame_2.astype(np.float32), states_1, states_2


# ═══════════════════════════════════════════════════════
# 主管線
# ═══════════════════════════════════════════════════════
def run_pipeline(enrollment_path: str, input_path: str, output_path: str,
                 threshold: float = 0.25,
                 gain_floor: float = 0.05,
                 attack_ms: float = 5.0,
                 release_ms: float = 50.0):
    """
    執行完整的離線管線。

    Parameters
    ----------
    enrollment_path : str
        enrollment 音頻路徑（目標說話者的乾淨語音）
    input_path : str
        混合音頻路徑
    output_path : str
        輸出音頻路徑
    threshold : float
        pVAD cosine similarity 閾值
    gain_floor : float
        最小增益（非目標部分的衰減）
    attack_ms, release_ms : float
        增益平滑的 attack / release 時間
    """
    print("\n" + "=" * 60)
    print("pVAD + SE 離線管線")
    print("=" * 60)

    # ── 1. 載入模型 ──────────────────────────────────
    print("\n[1/5] 載入模型...")
    # 優先使用 WeSpeaker ResNet34-LM（官方驗證的 ONNX，鑑別力更佳）
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        # 退而求其次用舊版 ECAPA-TDNN ONNX
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    assert ecapa_path.exists(), f"找不到 speaker encoder ONNX: {ecapa_path}"

    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    denoiser = DTLNDenoiser()
    gate = SoftGate(
        gain_floor=gain_floor,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    print("  ✅ 模型載入完成")

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

    # ── 4. 逐幀處理 ──────────────────────────────────
    print("\n[4/5] 逐幀處理...")
    pvad = SimplePVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        threshold=threshold,
    )

    # 切幀
    frames = frame_signal(input_audio)
    n_frames = frames.shape[0]
    print(f"  總幀數: {n_frames}")

    # 初始化輸出
    enhanced_frames = np.empty_like(frames)
    states_1, states_2 = None, None
    similarities = []

    t_start = time.time()

    for i in range(n_frames):
        # (a) DTLN 降噪
        denoised, states_1, states_2 = denoiser.process_frame(
            frames[i], states_1, states_2
        )

        # (b) pVAD：判定是否為目標說話者
        # 注意：這裡用降噪後的訊號做 pVAD（品質更好）
        is_target, sim = pvad.process_frame(denoised)
        similarities.append(sim)

        # (c) Soft gating
        enhanced_frames[i] = gate.process(denoised, is_target, confidence=sim)

        # 進度
        if (i + 1) % 500 == 0 or i == n_frames - 1:
            pct = (i + 1) / n_frames * 100
            avg_sim = np.mean(similarities[-100:]) if similarities else 0
            print(f"  [{pct:5.1f}%] frame {i+1}/{n_frames}, "
                  f"avg_sim={avg_sim:.3f}, target={is_target}")

    elapsed = time.time() - t_start
    rtf = elapsed / (len(input_audio) / SAMPLE_RATE)
    print(f"\n  處理時間: {elapsed:.2f}s, RTF: {rtf:.3f}")

    # ── 5. 合成輸出 ──────────────────────────────────
    print("\n[5/5] 合成輸出音頻...")
    output_audio = overlap_add(enhanced_frames)

    # 正規化（避免 clipping）
    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val
        print(f"  正規化: peak {max_val:.3f} → 0.99")

    write_audio(output_path, output_audio)
    print(f"  ✅ 輸出: {output_path}")

    # ── 統計 ──────────────────────────────────────────
    sims = np.array(similarities)
    print(f"\n📊 pVAD 統計:")
    print(f"  similarity 平均: {sims.mean():.3f}")
    print(f"  similarity 標準差: {sims.std():.3f}")
    print(f"  目標活躍幀比例: {(sims > threshold).mean():.1%}")
    print()


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="pVAD + SE 離線管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--enrollment", "-e", required=True,
        help="目標說話者的 enrollment 音頻路徑",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="混合音頻路徑",
    )
    parser.add_argument(
        "--output", "-o", default="output.wav",
        help="輸出音頻路徑（預設: output.wav）",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.25,
        help="pVAD cosine similarity 閾值（預設: 0.25）",
    )
    parser.add_argument(
        "--gain-floor", type=float, default=0.05,
        help="最小增益，即非目標部分的衰減量（預設: 0.05 = -26dB）",
    )
    parser.add_argument(
        "--attack-ms", type=float, default=5.0,
        help="增益上升時間 ms（預設: 5）",
    )
    parser.add_argument(
        "--release-ms", type=float, default=50.0,
        help="增益下降時間 ms（預設: 50）",
    )

    args = parser.parse_args()

    run_pipeline(
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
