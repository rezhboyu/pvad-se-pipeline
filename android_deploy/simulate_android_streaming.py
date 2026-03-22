#!/usr/bin/env python3
"""
模擬 Android 串流推論 Demo
============================
模擬 Android 端的完整即時處理流程：
1. 模擬麥克風輸入（每幀讀取 GTCRN_HOP=256 samples = 16ms）
2. GTCRN 逐幀降噪
3. CachedPVAD 定期檢查目標語者（每 0.5s 一次）
4. Soft gating 輸出
5. 模擬 AudioTrack 播放
6. 測量每幀延遲和總 RTF

用法:
    python simulate_android_streaming.py \\
        --enrollment enrollment.wav \\
        --input noisy_input.wav \\
        --output enhanced_output.wav

此腳本不依賴 Android 特定 API，純 Python + ONNX Runtime，
但邏輯完全對應 Android 端的 AudioRecord → 處理 → AudioTrack 流程。
"""

import argparse
import time
import sys
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path

# ══════════════════════════════════════════════════════════
# 常數（對應 Android 端設定）
# ══════════════════════════════════════════════════════════
SAMPLE_RATE = 16000
GTCRN_NFFT = 512
GTCRN_HOP = 256          # 16ms — Android AudioRecord 的讀取 buffer size
GTCRN_N_FREQ = 257
FBANK_N_MELS = 80
FBANK_WIN_LENGTH = 400   # 25ms
FBANK_HOP_LENGTH = 160   # 10ms
FBANK_N_FFT = 512

MODELS_DIR = Path(__file__).resolve().parent / "models"


# ══════════════════════════════════════════════════════════
# GTCRN 串流降噪器（模擬 Android JNI/ONNX）
# ══════════════════════════════════════════════════════════
class AndroidGTCRNDenoiser:
    """
    模擬 Android 端的 GTCRN 串流降噪。
    對應 Android 端會用 ONNX Runtime Mobile (onnxruntime-android)。
    """

    def __init__(self):
        model_path = MODELS_DIR / "gtcrn_simple.onnx"
        assert model_path.exists(), f"找不到 GTCRN: {model_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1  # Android 端通常用單執行緒

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.win = np.sqrt(np.hanning(GTCRN_NFFT)).astype(np.float32)
        self.input_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        self.output_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)
        self.win_sum_buffer = np.zeros(GTCRN_NFFT, dtype=np.float32)

        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

    def process(self, new_samples: np.ndarray) -> np.ndarray:
        """處理 256 個 samples，回傳 256 個增強 samples。"""
        # Shift input buffer
        self.input_buffer[:-GTCRN_HOP] = self.input_buffer[GTCRN_HOP:]
        self.input_buffer[-GTCRN_HOP:] = new_samples

        # STFT (one frame)
        windowed = self.input_buffer * self.win
        spec = np.fft.rfft(windowed, n=GTCRN_NFFT).astype(np.complex64)

        frame_input = np.zeros((1, 257, 1, 2), dtype=np.float32)
        frame_input[0, :, 0, 0] = spec.real
        frame_input[0, :, 0, 1] = spec.imag

        out, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            None, {
                'mix': frame_input,
                'conv_cache': self.conv_cache,
                'tra_cache': self.tra_cache,
                'inter_cache': self.inter_cache,
            }
        )

        enh_spec = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
        enh_frame = np.fft.irfft(enh_spec, n=GTCRN_NFFT).astype(np.float32)

        # OLA output
        win_sq = self.win ** 2
        out_norm = np.where(
            self.win_sum_buffer[:GTCRN_HOP] > 1e-8,
            self.output_buffer[:GTCRN_HOP] / self.win_sum_buffer[:GTCRN_HOP],
            0.0
        ).astype(np.float32)

        self.output_buffer[:-GTCRN_HOP] = self.output_buffer[GTCRN_HOP:]
        self.output_buffer[-GTCRN_HOP:] = 0.0
        self.win_sum_buffer[:-GTCRN_HOP] = self.win_sum_buffer[GTCRN_HOP:]
        self.win_sum_buffer[-GTCRN_HOP:] = 0.0

        self.output_buffer += enh_frame * self.win
        self.win_sum_buffer += win_sq

        return out_norm


# ══════════════════════════════════════════════════════════
# Speaker Encoder（模擬 Android 端的 enrollment + pVAD）
# ══════════════════════════════════════════════════════════
class AndroidSpeakerEncoder:
    """WeSpeaker ResNet34 ONNX 推論。"""

    def __init__(self):
        model_path = MODELS_DIR / "wespeaker_resnet34.onnx"
        assert model_path.exists(), f"找不到 WeSpeaker: {model_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """從音頻提取 L2-正規化的 d-vector。"""
        fbank = self._compute_fbank(audio)
        fbank_batch = fbank[np.newaxis, :, :]
        embedding = self.session.run(
            [self.output_name], {self.input_name: fbank_batch}
        )[0].squeeze()
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    @staticmethod
    def _compute_fbank(audio: np.ndarray) -> np.ndarray:
        """Kaldi-style Fbank (matching WeSpeaker training)."""
        import librosa
        stft = librosa.stft(
            audio, n_fft=FBANK_N_FFT, hop_length=FBANK_HOP_LENGTH,
            win_length=FBANK_WIN_LENGTH, window='hamming', center=False,
        )
        power_spec = np.abs(stft) ** 2
        mel_fb = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=FBANK_N_FFT, n_mels=FBANK_N_MELS,
            fmin=20, fmax=SAMPLE_RATE // 2,
        ).astype(np.float32)
        mel = mel_fb @ power_spec
        log_mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
        log_mel = log_mel - log_mel.mean(axis=1, keepdims=True)
        return log_mel.T.astype(np.float32)


# ══════════════════════════════════════════════════════════
# CachedPVAD（模擬 Android 端）
# ══════════════════════════════════════════════════════════
class AndroidCachedPVAD:
    """每 N 幀才提取一次 embedding 的 pVAD。"""

    def __init__(self, encoder, enrollment_dvector,
                 extract_interval=32, window_sec=0.5, threshold=0.25):
        self.encoder = encoder
        self.enrollment = enrollment_dvector
        self.threshold = threshold
        self.extract_interval = extract_interval
        self.window_samples = int(window_sec * SAMPLE_RATE)
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.frame_count = 0
        self._cached_sim = 0.0
        self._cached_is_target = False

    def process(self, samples):
        self.audio_buffer = np.concatenate([self.audio_buffer, samples])
        if len(self.audio_buffer) > self.window_samples:
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
        self.frame_count += 1
        if self.frame_count % self.extract_interval == 0:
            if len(self.audio_buffer) >= FBANK_WIN_LENGTH * 2:
                emb = self.encoder.extract_embedding(self.audio_buffer)
                self._cached_sim = float(np.dot(emb, self.enrollment))
                self._cached_is_target = self._cached_sim > self.threshold
        return self._cached_is_target, self._cached_sim


# ══════════════════════════════════════════════════════════
# Soft Gate
# ══════════════════════════════════════════════════════════
class AndroidSoftGate:
    def __init__(self, gain_floor=0.05, attack_ms=5.0, release_ms=50.0):
        self.gain_floor = gain_floor
        frame_dur = GTCRN_HOP / SAMPLE_RATE
        self.attack_coeff = 1.0 - np.exp(-frame_dur / max(attack_ms / 1000, 1e-6))
        self.release_coeff = 1.0 - np.exp(-frame_dur / max(release_ms / 1000, 1e-6))
        self.current_gain = gain_floor

    def process(self, samples, is_target, confidence):
        target_gain = self.gain_floor + (1.0 - self.gain_floor) * confidence if is_target else self.gain_floor
        coeff = self.attack_coeff if target_gain > self.current_gain else self.release_coeff
        self.current_gain += coeff * (target_gain - self.current_gain)
        return (samples * self.current_gain).astype(np.float32)


# ══════════════════════════════════════════════════════════
# Main Demo
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Android 串流推論模擬")
    parser.add_argument("--enrollment", "-e", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="output_android_sim.wav")
    parser.add_argument("--threshold", "-t", type=float, default=0.25)
    parser.add_argument("--pvad-interval", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print(" Android 串流推論模擬 Demo")
    print("=" * 60)

    # ── 初始化（對應 Android onCreate / onServiceConnected）──
    print("\n[INIT] 載入模型...")
    t_init = time.time()
    denoiser = AndroidGTCRNDenoiser()
    encoder = AndroidSpeakerEncoder()
    print(f"  GTCRN: {MODELS_DIR / 'gtcrn_simple.onnx'}")
    print(f"  WeSpeaker: {MODELS_DIR / 'wespeaker_resnet34.onnx'}")
    print(f"  模型載入耗時: {time.time() - t_init:.2f}s")

    # ── Enrollment（一次性，對應 Android 的 enrollment 頁面）──
    print("\n[ENROLLMENT] 提取目標語者 d-vector...")
    enroll_audio, sr = sf.read(args.enrollment, dtype='float32')
    if sr != SAMPLE_RATE:
        import librosa
        enroll_audio = librosa.resample(enroll_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if enroll_audio.ndim > 1:
        enroll_audio = enroll_audio[:, 0]
    t_enroll = time.time()
    enrollment_dvector = encoder.extract_embedding(enroll_audio)
    print(f"  d-vector dim: {enrollment_dvector.shape}, time: {time.time() - t_enroll:.3f}s")

    # ── 建立 pVAD + gate ──
    pvad = AndroidCachedPVAD(
        encoder, enrollment_dvector,
        extract_interval=args.pvad_interval,
        threshold=args.threshold,
    )
    gate = AndroidSoftGate()

    # ── 模擬 AudioRecord 輸入 ──
    print("\n[STREAMING] 模擬麥克風輸入...")
    input_audio, sr = sf.read(args.input, dtype='float32')
    if sr != SAMPLE_RATE:
        import librosa
        input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if input_audio.ndim > 1:
        input_audio = input_audio[:, 0]

    total_samples = len(input_audio)
    total_shifts = total_samples // GTCRN_HOP
    duration = total_samples / SAMPLE_RATE
    print(f"  輸入: {duration:.2f}s, {total_shifts} frames ({GTCRN_HOP/SAMPLE_RATE*1000:.0f}ms/frame)")

    # ── 串流處理迴圈（對應 Android AudioRecord callback）──
    output_chunks = []
    timings = {'total': [], 'denoise': [], 'pvad': [], 'gate': []}

    padded_len = total_shifts * GTCRN_HOP
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[:min(total_samples, padded_len)] = input_audio[:padded_len]

    for i in range(total_shifts):
        chunk = padded[i * GTCRN_HOP:(i + 1) * GTCRN_HOP]
        t0 = time.perf_counter()

        # Step 1: GTCRN denoise
        t1 = time.perf_counter()
        denoised = denoiser.process(chunk)
        t2 = time.perf_counter()

        # Step 2: pVAD
        is_target, sim = pvad.process(denoised)
        t3 = time.perf_counter()

        # Step 3: Soft gate
        output = gate.process(denoised, is_target, sim)
        t4 = time.perf_counter()

        output_chunks.append(output)
        timings['denoise'].append(t2 - t1)
        timings['pvad'].append(t3 - t2)
        timings['gate'].append(t4 - t3)
        timings['total'].append(t4 - t0)

        # Progress
        fps = SAMPLE_RATE // GTCRN_HOP
        if (i + 1) % fps == 0:
            sec = (i + 1) * GTCRN_HOP / SAMPLE_RATE
            avg = np.mean(timings['total'][-fps:]) * 1000
            print(f"  [{sec:5.1f}s/{duration:.1f}s] frame_time={avg:.2f}ms, sim={sim:.3f}")

    # ── 輸出（對應 Android AudioTrack）──
    output_audio = np.concatenate(output_chunks)[:total_samples]
    peak = np.max(np.abs(output_audio))
    if peak > 0.99:
        output_audio *= 0.99 / peak

    sf.write(args.output, output_audio, SAMPLE_RATE)
    print(f"\n  輸出: {args.output}")

    # ── 效能報告 ──
    frame_ms = GTCRN_HOP / SAMPLE_RATE * 1000  # 16ms
    total_ms = np.array(timings['total']) * 1000
    denoise_ms = np.array(timings['denoise']) * 1000
    pvad_ms = np.array(timings['pvad']) * 1000
    gate_ms = np.array(timings['gate']) * 1000

    report = {
        'frame_duration_ms': frame_ms,
        'total': {
            'mean_ms': float(total_ms.mean()),
            'p50_ms': float(np.median(total_ms)),
            'p95_ms': float(np.percentile(total_ms, 95)),
            'p99_ms': float(np.percentile(total_ms, 99)),
            'max_ms': float(total_ms.max()),
            'rtf': float(total_ms.mean() / frame_ms),
        },
        'denoise_gtcrn': {
            'mean_ms': float(denoise_ms.mean()),
            'p95_ms': float(np.percentile(denoise_ms, 95)),
        },
        'pvad_cached': {
            'mean_ms': float(pvad_ms.mean()),
            'p95_ms': float(np.percentile(pvad_ms, 95)),
            'extract_interval': args.pvad_interval,
        },
        'gate': {
            'mean_ms': float(gate_ms.mean()),
        },
    }

    print(f"\n{'='*60}")
    print(" 效能報告（模擬 Android CPU 推論）")
    print(f"{'='*60}")
    print(f"  即時約束: {frame_ms:.0f}ms per frame")
    print(f"  GTCRN 降噪: {denoise_ms.mean():.2f}ms avg, {np.percentile(denoise_ms, 95):.2f}ms P95")
    print(f"  pVAD (cached): {pvad_ms.mean():.2f}ms avg, {np.percentile(pvad_ms, 95):.2f}ms P95")
    print(f"  Gate: {gate_ms.mean():.4f}ms avg")
    print(f"  總計: {total_ms.mean():.2f}ms avg, RTF={total_ms.mean()/frame_ms:.4f}")
    print()

    if total_ms.mean() / frame_ms < 1.0:
        print(f"  結論: RTF={total_ms.mean()/frame_ms:.4f} < 1.0, 可即時處理")
        print(f"  Android 端 CPU 預估 RTF (2-3x slower): {total_ms.mean()/frame_ms * 2.5:.3f}")
    else:
        print(f"  結論: RTF={total_ms.mean()/frame_ms:.4f} >= 1.0, 需要優化")

    # 儲存報告
    report_path = Path(args.output).parent / "performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  效能報告: {report_path}")


if __name__ == "__main__":
    main()
