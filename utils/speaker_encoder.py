"""
Speaker Encoder 封裝
====================
封裝 WeSpeaker ResNet34 的 ONNX 推論，用於：
1. 從 enrollment 音頻提取 d-vector
2. 從短窗音頻提取即時 embedding（供 pVAD 用）
3. 計算 cosine similarity

模型：WeSpeaker ResNet34-LM (voxceleb_resnet34_LM.onnx)
  - 輸入：Kaldi-style Fbank (batch, T, 80)
  - 輸出：speaker embedding (batch, 256)
  - 官方驗證的 ONNX，無需自行匯出

注意：此模組只用 ONNX Runtime + librosa，不依賴 PyTorch。
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from .audio import SAMPLE_RATE, BLOCK_LEN, BLOCK_SHIFT, read_audio

# ── WeSpeaker Fbank 參數（Kaldi-compatible）──────────────
FBANK_N_MELS = 80
FBANK_WIN_LENGTH = 400    # 25ms at 16kHz
FBANK_HOP_LENGTH = 160    # 10ms at 16kHz
FBANK_N_FFT = 512         # next power of 2


def _compute_fbank(audio: np.ndarray) -> np.ndarray:
    """
    計算 Kaldi-style Fbank 特徵（匹配 WeSpeaker 訓練時的特徵提取）。

    Parameters
    ----------
    audio : (n_samples,) float32, 16kHz mono

    Returns
    -------
    fbank : (T, 80) float32, 已做 sentence-level CMN
    """
    import librosa

    # STFT: hamming window, no center padding (Kaldi default)
    stft = librosa.stft(
        audio, n_fft=FBANK_N_FFT, hop_length=FBANK_HOP_LENGTH,
        win_length=FBANK_WIN_LENGTH, window='hamming', center=False,
    )
    # Power spectrum
    power_spec = np.abs(stft) ** 2

    # Mel filterbank
    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=FBANK_N_FFT, n_mels=FBANK_N_MELS,
        fmin=20, fmax=SAMPLE_RATE // 2,
    ).astype(np.float32)

    mel = mel_fb @ power_spec  # (n_mels, T)
    log_mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)

    # Sentence-level CMN (Cepstral Mean Normalization)
    log_mel = log_mel - log_mel.mean(axis=1, keepdims=True)

    return log_mel.T.astype(np.float32)  # (T, n_mels)


class SpeakerEncoder:
    """
    WeSpeaker ResNet34-LM ONNX 推論封裝。

    模型輸入：Fbank 特徵 (batch, T, 80)
    模型輸出：speaker embedding (batch, 256)
    """

    def __init__(self, onnx_path: str, n_mels: int = FBANK_N_MELS):
        self.onnx_path = Path(onnx_path)
        assert self.onnx_path.exists(), f"ONNX 模型不存在：{onnx_path}"

        # 建立 ONNX Runtime session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        # 取得輸入 / 輸出名稱
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.n_mels = n_mels

        # 取得 embedding 維度
        out_shape = self.session.get_outputs()[0].shape
        self.embed_dim = out_shape[-1] if isinstance(out_shape[-1], int) else 256

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        從時域音頻提取 speaker embedding。
        輸入: (n_samples,) float32, 16kHz
        輸出: (embed_dim,) float32，已 L2 正規化
        """
        # 計算 Fbank 特徵
        fbank = _compute_fbank(audio)  # (T, n_mels)

        # 加 batch 維度：(1, T, n_mels)
        fbank_batch = fbank[np.newaxis, :, :]

        # ONNX 推論
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: fbank_batch}
        )[0]  # (1, embed_dim) or (1, 1, embed_dim)

        embedding = embedding.squeeze()  # (embed_dim,)

        # L2 正規化
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def extract_embedding_from_file(self, audio_path: str) -> np.ndarray:
        """從音頻檔提取 speaker embedding。"""
        audio = read_audio(audio_path)
        return self.extract_embedding(audio)

    def extract_augmented_embedding(
        self,
        audio: np.ndarray,
        augmentations: list[tuple[str, float]] | None = None,
        normalize_centroid: bool = True,
    ) -> np.ndarray:
        """
        對 enrollment 音頻做多種增強，取 embedding centroid。

        Parameters
        ----------
        audio : (n_samples,) float32, 16kHz
        augmentations : list of (noise_type, snr_db)
            noise_type: "white", "pink", "none"(原始)
            snr_db: 信噪比 (dB)，noise_type="none" 時忽略
            預設: 原始 + 白噪 20/10/5 dB + 粉紅噪 10 dB
        normalize_centroid : bool
            True: L2 normalize centroid（推薦），False: 只取平均

        Returns
        -------
        centroid : (embed_dim,) float32，已 L2 正規化
        """
        if augmentations is None:
            augmentations = [
                ("none", 0),
                ("white", 20),
                ("white", 10),
                ("white", 5),
                ("pink", 10),
            ]

        embeddings = []
        for noise_type, snr_db in augmentations:
            aug_audio = self._apply_noise(audio, noise_type, snr_db)
            emb = self.extract_embedding(aug_audio)
            embeddings.append(emb)

        # Centroid: 平均後 L2 normalize
        centroid = np.mean(embeddings, axis=0)
        if normalize_centroid:
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid = centroid / norm
        return centroid.astype(np.float32)

    def extract_augmented_embedding_from_file(
        self, audio_path: str, **kwargs
    ) -> np.ndarray:
        """從音頻檔提取 augmented enrollment embedding。"""
        audio = read_audio(audio_path)
        return self.extract_augmented_embedding(audio, **kwargs)

    @staticmethod
    def _apply_noise(
        audio: np.ndarray, noise_type: str, snr_db: float
    ) -> np.ndarray:
        """對音頻加入指定類型和 SNR 的噪音。"""
        if noise_type == "none":
            return audio.copy()

        rms_signal = np.sqrt(np.mean(audio ** 2) + 1e-12)

        if noise_type == "white":
            noise = np.random.randn(len(audio)).astype(np.float32)
        elif noise_type == "pink":
            noise = SpeakerEncoder._generate_pink_noise(len(audio))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        rms_noise = np.sqrt(np.mean(noise ** 2) + 1e-12)
        scale = rms_signal / (rms_noise * 10 ** (snr_db / 20))
        noisy = audio + noise * scale

        # Clip to prevent overflow
        peak = np.max(np.abs(noisy))
        if peak > 0.99:
            noisy = noisy * 0.99 / peak
        return noisy.astype(np.float32)

    @staticmethod
    def _generate_pink_noise(n_samples: int) -> np.ndarray:
        """生成粉紅噪音（1/f spectrum）。"""
        white = np.random.randn(n_samples).astype(np.float32)
        # 用 FFT 做 1/f 濾波
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-6  # 避免 DC 除零
        spectrum = np.fft.rfft(white)
        pink_filter = 1.0 / np.sqrt(freqs)
        spectrum *= pink_filter
        pink = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
        return pink


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    計算兩個向量的 cosine similarity。
    假設輸入已經 L2 正規化，直接做內積。
    """
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b))


class SimplePVAD:
    """
    簡易 pVAD（原始版本，每幀都提取 embedding — 僅供向後相容）
    """

    def __init__(self, speaker_encoder: SpeakerEncoder,
                 enrollment_dvector: np.ndarray,
                 window_sec: float = 0.5,
                 threshold: float = 0.25):
        self.encoder = speaker_encoder
        self.enrollment = enrollment_dvector
        self.threshold = threshold
        self.window_samples = int(window_sec * SAMPLE_RATE)
        self.window_frames = max(1, int(window_sec * SAMPLE_RATE / BLOCK_SHIFT))
        self.audio_buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self.audio_buffer = np.zeros(0, dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> tuple:
        self.audio_buffer = np.concatenate([self.audio_buffer, frame])
        if len(self.audio_buffer) > self.window_samples:
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
        min_samples = FBANK_WIN_LENGTH * 2
        if len(self.audio_buffer) < min_samples:
            return False, 0.0
        embedding = self.encoder.extract_embedding(self.audio_buffer)
        sim = cosine_similarity(embedding, self.enrollment)
        return sim > self.threshold, sim


class CachedPVAD:
    """
    快取版 pVAD — 每 N 幀才跑一次 WeSpeaker embedding 提取
    ==========================================================
    解決 SimplePVAD 的 RTF 問題：原本每幀都跑 WeSpeaker (RTF≈18)，
    改成每 extract_interval 幀才跑一次，中間沿用上次的 similarity 值。

    在 16ms hop (GTCRN) 下，extract_interval=32 表示每 0.5 秒提取一次。

    Parameters
    ----------
    speaker_encoder : SpeakerEncoder
        ONNX speaker encoder
    enrollment_dvector : np.ndarray
        enrollment 的 d-vector (embed_dim,), 已 L2 正規化
    extract_interval : int
        每幾幀才跑一次 embedding 提取（預設 32 = 0.5s at 16ms hop）
    window_sec : float
        提取 embedding 時用的滑窗長度（秒）
    threshold : float
        cosine similarity 閾值
    """

    def __init__(self, speaker_encoder: SpeakerEncoder,
                 enrollment_dvector: np.ndarray,
                 extract_interval: int = 32,
                 window_sec: float = 0.5,
                 threshold: float = 0.25):
        self.encoder = speaker_encoder
        self.enrollment = enrollment_dvector
        self.threshold = threshold
        self.extract_interval = extract_interval
        self.window_samples = int(window_sec * SAMPLE_RATE)

        # 音頻緩衝區
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        # 幀計數器
        self.frame_count = 0
        # 快取結果
        self._cached_sim = 0.0
        self._cached_is_target = False

    def reset(self) -> None:
        """重設所有內部狀態。"""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.frame_count = 0
        self._cached_sim = 0.0
        self._cached_is_target = False

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        處理一幀，回傳 (is_target, similarity)。

        只有在 frame_count % extract_interval == 0 時才真正跑 embedding 提取，
        其餘幀直接回傳上次的結果。

        Parameters
        ----------
        frame : 時域音頻 samples

        Returns
        -------
        is_target : bool
        similarity : float
        """
        # 累積到緩衝區
        self.audio_buffer = np.concatenate([self.audio_buffer, frame])
        if len(self.audio_buffer) > self.window_samples:
            self.audio_buffer = self.audio_buffer[-self.window_samples:]

        self.frame_count += 1

        # 判斷是否該跑 embedding
        should_extract = (self.frame_count % self.extract_interval == 0)

        if should_extract:
            min_samples = FBANK_WIN_LENGTH * 2
            if len(self.audio_buffer) >= min_samples:
                embedding = self.encoder.extract_embedding(self.audio_buffer)
                self._cached_sim = cosine_similarity(embedding, self.enrollment)
                self._cached_is_target = self._cached_sim > self.threshold

        return self._cached_is_target, self._cached_sim
