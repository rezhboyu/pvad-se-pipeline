"""
Personal VAD 推論模組
====================
基於 ONNX Runtime 的 frame-level 目標說話者偵測。

用法:
    pvad = PersonalVAD("models/personal_vad/personal_vad.onnx", enrollment_dvector)
    for frame in audio_frames:
        is_target, confidence = pvad.process_frame(frame)
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

from .speaker_encoder import _compute_fbank, FBANK_HOP_LENGTH, FBANK_WIN_LENGTH
from .audio import SAMPLE_RATE


class PersonalVAD:
    """
    Personal VAD: frame-level 目標說話者偵測。

    輸入 Fbank + enrollment d-vector，輸出逐幀 3 類機率：
      [non_speech, target, non_target]

    Parameters
    ----------
    onnx_path : str
        Personal VAD ONNX 模型路徑
    enrollment_dvector : np.ndarray
        (embed_dim,) L2-normalized enrollment d-vector
    """

    # Label indices
    NON_SPEECH = 0
    TARGET = 1
    NON_TARGET = 2

    def __init__(self, onnx_path: str, enrollment_dvector: np.ndarray):
        self.onnx_path = Path(onnx_path)
        assert self.onnx_path.exists(), f"ONNX 模型不存在: {onnx_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.enrollment_dvector = enrollment_dvector.astype(np.float32)
        self.embed_dim = len(enrollment_dvector)

        # Streaming state
        self.audio_buffer = np.array([], dtype=np.float32)
        self._last_is_target = False
        self._last_confidence = 0.0
        self._frame_count = 0

        # 計算需要多少 audio samples 產生一個 Fbank frame
        self._hop = FBANK_HOP_LENGTH   # 160 samples = 10ms
        self._win = FBANK_WIN_LENGTH   # 400 samples = 25ms

    def process_frame(self, audio_samples: np.ndarray):
        """
        處理一段音訊（通常是 GTCRN hop = 256 samples = 16ms）。

        Parameters
        ----------
        audio_samples : np.ndarray
            (n_samples,) float32 raw audio

        Returns
        -------
        is_target : bool
        confidence : float (target class probability)
        """
        # 累積音訊
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_samples])

        # 計算可以產生多少新的 Fbank frames
        n_available = len(self.audio_buffer)
        n_fbank_frames = max(0, (n_available - self._win) // self._hop + 1)

        if n_fbank_frames == 0:
            return self._last_is_target, self._last_confidence

        # 計算 Fbank（整個 buffer）
        fbank = _compute_fbank(self.audio_buffer)  # (T, 80)

        if len(fbank) == 0:
            return self._last_is_target, self._last_confidence

        # 拼接 d-vector
        n_frames = len(fbank)
        dvector_rep = np.tile(self.enrollment_dvector, (n_frames, 1))  # (T, embed_dim)
        features = np.concatenate([fbank, dvector_rep], axis=1)  # (T, 80+embed_dim)

        # ONNX 推論
        features_batch = features[np.newaxis, :, :].astype(np.float32)  # (1, T, D)
        logits = self.session.run(None, {"features": features_batch})[0]  # (1, T, 3)

        # 取最後一幀的結果（最新的判定）
        last_logits = logits[0, -1, :]  # (3,)

        # Softmax
        exp_logits = np.exp(last_logits - np.max(last_logits))
        probs = exp_logits / exp_logits.sum()

        self._last_confidence = float(probs[self.TARGET])
        self._last_is_target = bool(probs[self.TARGET] > probs[self.NON_TARGET]
                                     and probs[self.TARGET] > probs[self.NON_SPEECH])

        # 保留最近的音訊（避免 buffer 無限增長）
        # 保留最後 1 秒的音訊供 LSTM context
        max_buffer = SAMPLE_RATE  # 1 秒
        if len(self.audio_buffer) > max_buffer:
            self.audio_buffer = self.audio_buffer[-max_buffer:]

        self._frame_count += 1
        return self._last_is_target, self._last_confidence

    def reset(self):
        """重置 streaming state。"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self._last_is_target = False
        self._last_confidence = 0.0
        self._frame_count = 0
