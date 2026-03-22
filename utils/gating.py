"""
Soft Gating 模組
================
根據 pVAD 的活躍度分數，對降噪後的訊號施加平滑的增益控制。
非目標說話者的部分不會被完全靜音，而是衰減至 gain_floor。

參數說明
--------
- gain_floor : 最小增益（0.05 = -26 dB），避免完全靜音造成的不自然感
- attack_ms : 增益從低到高的上升時間（毫秒）
- release_ms : 增益從高到低的下降時間（毫秒）
"""

import numpy as np
from .audio import SAMPLE_RATE, BLOCK_SHIFT


class SoftGate:
    """
    帶 attack / release 平滑的 soft gate。

    使用方式：
        gate = SoftGate(gain_floor=0.05, attack_ms=5, release_ms=50)
        for each frame:
            gated_frame = gate.process(frame, is_target_active)
    """

    def __init__(self, gain_floor: float = 0.05,
                 attack_ms: float = 5.0,
                 release_ms: float = 50.0,
                 sr: int = SAMPLE_RATE,
                 hop: int = BLOCK_SHIFT):
        self.gain_floor = gain_floor

        # 將 attack / release 時間轉換為每幀的平滑係數
        # 一幀的時間 = hop / sr 秒
        frame_duration_s = hop / sr

        # 平滑係數：alpha = 1 - exp(-frame_duration / tau)
        # tau = time_constant ≈ attack_ms / 1000 (以秒計)
        attack_tau = max(attack_ms / 1000.0, 1e-6)
        release_tau = max(release_ms / 1000.0, 1e-6)

        self.attack_coeff = 1.0 - np.exp(-frame_duration_s / attack_tau)
        self.release_coeff = 1.0 - np.exp(-frame_duration_s / release_tau)

        # 目前的平滑增益值（初始化為 gain_floor，代表靜默起始）
        self.current_gain = gain_floor

    def reset(self) -> None:
        """重設內部狀態。"""
        self.current_gain = self.gain_floor

    def compute_target_gain(self, is_target: bool, confidence: float = 1.0) -> float:
        """
        計算目標增益。

        Parameters
        ----------
        is_target : bool
            pVAD 判定是否為目標說話者
        confidence : float
            pVAD 信心度 (0~1)，可用於軟性調控

        Returns
        -------
        target_gain : float
            目標增益值 (gain_floor ~ 1.0)
        """
        if is_target:
            # 目標活躍：增益 = gain_floor + (1 - gain_floor) * confidence
            return self.gain_floor + (1.0 - self.gain_floor) * confidence
        else:
            return self.gain_floor

    def smooth_gain(self, target_gain: float) -> float:
        """
        對目標增益做一階 IIR 平滑（attack / release 不對稱）。
        """
        if target_gain > self.current_gain:
            # 上升 → 用 attack 係數
            coeff = self.attack_coeff
        else:
            # 下降 → 用 release 係數
            coeff = self.release_coeff

        self.current_gain += coeff * (target_gain - self.current_gain)
        return self.current_gain

    def process(self, frame: np.ndarray, is_target: bool,
                confidence: float = 1.0) -> np.ndarray:
        """
        處理一幀音頻（滑動窗口：幀內逐 sample 線性內插增益）。

        Parameters
        ----------
        frame : np.ndarray, shape (block_len,)
            輸入音頻幀
        is_target : bool
            此幀是否為目標說話者
        confidence : float
            pVAD 信心度

        Returns
        -------
        gated_frame : np.ndarray
            經增益控制後的音頻幀
        """
        prev_gain = self.current_gain
        target_gain = self.compute_target_gain(is_target, confidence)
        new_gain = self.smooth_gain(target_gain)

        # 幀內逐 sample 線性內插，避免增益跳變
        n = len(frame)
        gain_ramp = np.linspace(prev_gain, new_gain, n, dtype=np.float32)
        return (frame * gain_ramp).astype(np.float32)

    def process_batch(self, frames: np.ndarray,
                      is_target_flags: np.ndarray,
                      confidences: np.ndarray = None) -> np.ndarray:
        """
        批次處理多幀（離線用）。

        Parameters
        ----------
        frames : (n_frames, block_len)
        is_target_flags : (n_frames,) bool
        confidences : (n_frames,) float, optional

        Returns
        -------
        gated_frames : (n_frames, block_len)
        """
        n_frames = frames.shape[0]
        if confidences is None:
            confidences = np.ones(n_frames, dtype=np.float32)

        output = np.empty_like(frames)
        for i in range(n_frames):
            output[i] = self.process(frames[i], bool(is_target_flags[i]),
                                     float(confidences[i]))
        return output
