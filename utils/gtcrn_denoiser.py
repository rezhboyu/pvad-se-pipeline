"""
GTCRN 降噪器封裝
================
GTCRN (Group-Temporal Convolutional Recurrent Network) ONNX 推論封裝。
支援離線逐幀處理和串流模式。

模型規格：
  - nfft = 512, hop = 256 (16ms at 16kHz)
  - 輸入: mix (1, 257, 1, 2) 即 real + imag of one STFT frame
  - 快取: conv_cache (2,1,16,16,33), tra_cache (2,3,1,1,16), inter_cache (2,1,33,16)
  - 輸出: enh (1, 257, 1, 2) + updated caches
  - 模型大小: ~535KB, 非常適合行動裝置
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# GTCRN 專用常數
GTCRN_NFFT = 512
GTCRN_HOP = 256          # 16ms at 16kHz
GTCRN_N_FREQ = GTCRN_NFFT // 2 + 1  # 257


def _sqrt_hann(length: int) -> np.ndarray:
    """sqrt-Hann window (matching GTCRN training)."""
    return np.sqrt(np.hanning(length)).astype(np.float32)


class GTCRNDenoiser:
    """
    GTCRN 離線降噪器。

    一次處理整段音頻（內部逐幀 + overlap-add）。
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parent.parent / "models" / "gtcrn" / "gtcrn_simple.onnx"
            )
        model_path = Path(model_path)
        assert model_path.exists(), f"找不到 GTCRN ONNX: {model_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.win = _sqrt_hann(GTCRN_NFFT)
        print(f"[GTCRN] 模型載入完成: {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB)")

    def _init_caches(self):
        conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)
        return conv_cache, tra_cache, inter_cache

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        離線增強整段音頻。

        Parameters
        ----------
        audio : (n_samples,) float32, 16kHz

        Returns
        -------
        enhanced : (n_samples,) float32
        """
        orig_len = len(audio)
        pad_len = GTCRN_HOP - (orig_len % GTCRN_HOP) if orig_len % GTCRN_HOP != 0 else 0
        audio_padded = np.concatenate([audio, np.zeros(pad_len, dtype=np.float32)])

        num_frames = (len(audio_padded) - GTCRN_NFFT) // GTCRN_HOP + 1

        # 切幀 + 加窗 + FFT
        from numpy.fft import rfft, irfft

        frames = np.zeros((num_frames, GTCRN_NFFT), dtype=np.float32)
        for i in range(num_frames):
            start = i * GTCRN_HOP
            frames[i] = audio_padded[start:start + GTCRN_NFFT] * self.win

        spec = rfft(frames, n=GTCRN_NFFT, axis=-1)  # (T, 257)

        # 初始化快取
        conv_cache, tra_cache, inter_cache = self._init_caches()

        outputs_real = []
        outputs_imag = []

        for i in range(num_frames):
            frame_input = np.zeros((1, 257, 1, 2), dtype=np.float32)
            frame_input[0, :, 0, 0] = spec[i].real
            frame_input[0, :, 0, 1] = spec[i].imag

            out, conv_cache, tra_cache, inter_cache = self.session.run(
                None, {
                    'mix': frame_input,
                    'conv_cache': conv_cache,
                    'tra_cache': tra_cache,
                    'inter_cache': inter_cache,
                }
            )
            outputs_real.append(out[0, :, 0, 0])
            outputs_imag.append(out[0, :, 0, 1])

        # 重建
        enh_spec = np.array(outputs_real) + 1j * np.array(outputs_imag)
        enh_frames = irfft(enh_spec, n=GTCRN_NFFT, axis=-1).astype(np.float32)

        # Overlap-add with window
        output = np.zeros(len(audio_padded), dtype=np.float32)
        win_sum = np.zeros(len(audio_padded), dtype=np.float32)

        for i in range(num_frames):
            start = i * GTCRN_HOP
            output[start:start + GTCRN_NFFT] += enh_frames[i] * self.win
            win_sum[start:start + GTCRN_NFFT] += self.win ** 2

        win_sum = np.maximum(win_sum, 1e-8)
        output = output / win_sum

        return output[:orig_len]


class StreamingGTCRNDenoiser:
    """
    串流版 GTCRN 降噪器。

    每次推入 GTCRN_HOP (256) 個新 samples，回傳 GTCRN_HOP 個增強 samples。
    內部維護輸入緩衝區和 ONNX cache states。
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parent.parent / "models" / "gtcrn" / "gtcrn_simple.onnx"
            )
        model_path = Path(model_path)
        assert model_path.exists(), f"找不到 GTCRN ONNX: {model_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1  # 串流版用單執行緒降低延遲

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.win = _sqrt_hann(GTCRN_NFFT)

        # 輸入緩衝區
        self.input_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        # 輸出 overlap-add 緩衝區
        self.output_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        # window sum for normalization
        self.win_sum_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)

        # ONNX caches
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

    def reset(self):
        """重設所有內部狀態。"""
        self.input_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        self.output_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        self.win_sum_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

    def process_shift(self, new_samples: np.ndarray) -> np.ndarray:
        """
        推入 GTCRN_HOP (256) 個新 samples，回傳 GTCRN_HOP 個增強 samples。

        Parameters
        ----------
        new_samples : (GTCRN_HOP,) float32

        Returns
        -------
        output_samples : (GTCRN_HOP,) float32
        """
        assert len(new_samples) == GTCRN_HOP, \
            f"期望 {GTCRN_HOP} 個 samples，得到 {len(new_samples)}"

        # 移動輸入緩衝區，加入新 samples
        self.input_buffer[:-GTCRN_HOP] = self.input_buffer[GTCRN_HOP:]
        self.input_buffer[-GTCRN_HOP:] = new_samples

        # 加窗 + FFT
        windowed = self.input_buffer * self.win
        spec = np.fft.rfft(windowed, n=GTCRN_NFFT).astype(np.complex64)

        # 準備 ONNX 輸入
        frame_input = np.zeros((1, 257, 1, 2), dtype=np.float32)
        frame_input[0, :, 0, 0] = spec.real
        frame_input[0, :, 0, 1] = spec.imag

        # 推論
        out, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            None, {
                'mix': frame_input,
                'conv_cache': self.conv_cache,
                'tra_cache': self.tra_cache,
                'inter_cache': self.inter_cache,
            }
        )

        # 重建時域
        enh_spec = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
        enh_frame = np.fft.irfft(enh_spec, n=GTCRN_NFFT).astype(np.float32)

        # 取出最早的 GTCRN_HOP 個 samples 作為輸出（在 OLA 正規化後）
        # 先從 output_buffer 取出
        win_sq = self.win ** 2
        ws = np.maximum(self.win_sum_buffer[:GTCRN_HOP], 1e-8)
        out_norm = (self.output_buffer[:GTCRN_HOP] / ws).astype(np.float32)

        # 移動緩衝區
        self.output_buffer[:-GTCRN_HOP] = self.output_buffer[GTCRN_HOP:]
        self.output_buffer[-GTCRN_HOP:] = 0.0
        self.win_sum_buffer[:-GTCRN_HOP] = self.win_sum_buffer[GTCRN_HOP:]
        self.win_sum_buffer[-GTCRN_HOP:] = 0.0

        # 加入新的增強幀（加窗）
        self.output_buffer += enh_frame * self.win
        self.win_sum_buffer += win_sq

        return out_norm
