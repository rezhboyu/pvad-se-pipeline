"""
音頻工具模組
============
提供音頻 I/O、STFT / ISTFT、Mel 濾波器組等基礎功能。
所有操作僅依賴 numpy / scipy / soundfile，不需要 PyTorch。
"""

import numpy as np
import soundfile as sf
from scipy.signal import get_window

# ── 常數 ──────────────────────────────────────────────
SAMPLE_RATE = 16_000          # 取樣率 16 kHz
BLOCK_LEN = 512               # 每幀樣本數（32 ms）
BLOCK_SHIFT = 128             # 幀移（8 ms）
FFT_SIZE = 512                # FFT 長度
N_FREQ_BINS = FFT_SIZE // 2 + 1   # 頻率 bin 數 = 257
N_MEL = 80                    # Mel 濾波器數量


# ── 音頻 I/O ──────────────────────────────────────────
def read_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """讀取音頻檔並重新取樣至目標取樣率，回傳 float32 mono 訊號。"""
    data, file_sr = sf.read(path, dtype="float32", always_2d=False)
    # 如果是多聲道，取第一聲道
    if data.ndim > 1:
        data = data[:, 0]
    # 簡易重新取樣（整數比率）— 若需高品質可改用 librosa.resample
    if file_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data.astype(np.float32)


def write_audio(path: str, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """將 float32 音頻寫入 wav 檔。"""
    sf.write(path, data, sr, subtype="FLOAT")


# ── STFT / ISTFT ──────────────────────────────────────
def _analysis_window() -> np.ndarray:
    """回傳分析窗（Hann 窗），長度 = BLOCK_LEN。"""
    return get_window("hann", BLOCK_LEN, fftbins=True).astype(np.float32)


def stft_frame(frame: np.ndarray) -> np.ndarray:
    """
    對單一幀做 STFT。
    輸入: (BLOCK_LEN,) 時域訊號
    輸出: (N_FREQ_BINS,) 複數頻譜
    """
    win = _analysis_window()
    return np.fft.rfft(frame * win, n=FFT_SIZE).astype(np.complex64)


def istft_frame(spectrum: np.ndarray) -> np.ndarray:
    """
    對單一幀做 ISTFT（overlap-add 的一幀部分）。
    輸入: (N_FREQ_BINS,) 複數頻譜
    輸出: (BLOCK_LEN,) 時域訊號
    """
    win = _analysis_window()
    return (np.fft.irfft(spectrum, n=FFT_SIZE)[:BLOCK_LEN] * win).astype(np.float32)


def magnitude_phase(spectrum: np.ndarray):
    """分離幅度與相位。"""
    mag = np.abs(spectrum).astype(np.float32)
    phase = np.angle(spectrum).astype(np.float32)
    return mag, phase


def reconstruct_complex(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """從幅度和相位重建複數頻譜。"""
    return (mag * np.exp(1j * phase)).astype(np.complex64)


# ── Mel 濾波器組 ──────────────────────────────────────
def mel_filterbank(n_mels: int = N_MEL, n_fft: int = FFT_SIZE,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    計算 Mel 濾波器組矩陣。
    回傳: (n_mels, n_fft//2+1) float32
    """
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)


def log_mel_spectrogram(mag: np.ndarray, mel_fb: np.ndarray) -> np.ndarray:
    """
    從幅度頻譜計算 log-Mel 特徵。
    mag: (N_FREQ_BINS,) or (T, N_FREQ_BINS)
    mel_fb: (n_mels, N_FREQ_BINS)
    回傳: 同形狀但頻率軸變為 n_mels
    """
    if mag.ndim == 1:
        mel = mel_fb @ mag
    else:
        mel = (mel_fb @ mag.T).T
    # log 壓縮，加小常數避免 log(0)
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


# ── 幀切割 ────────────────────────────────────────────
def frame_signal(signal: np.ndarray, block_len: int = BLOCK_LEN,
                 block_shift: int = BLOCK_SHIFT) -> np.ndarray:
    """
    將整段訊號切成重疊幀。
    回傳: (n_frames, block_len)
    """
    n_frames = (len(signal) - block_len) // block_shift + 1
    indices = np.arange(block_len)[None, :] + np.arange(n_frames)[:, None] * block_shift
    return signal[indices].astype(np.float32)


def overlap_add(frames: np.ndarray, block_len: int = BLOCK_LEN,
                block_shift: int = BLOCK_SHIFT) -> np.ndarray:
    """
    Overlap-add 合成。
    frames: (n_frames, block_len)
    回傳: 一維時域訊號
    """
    n_frames = frames.shape[0]
    output_len = (n_frames - 1) * block_shift + block_len
    output = np.zeros(output_len, dtype=np.float32)
    for i in range(n_frames):
        start = i * block_shift
        output[start:start + block_len] += frames[i]
    return output
