#!/usr/bin/env python3
"""
並行版管線（Parallel pVAD + GTCRN SE）
=======================================
與串行版（pipeline_gtcrn.py）的關鍵差異：
  串行：raw → GTCRN → denoised → pVAD → gating → output
  並行：raw → ┬─ GTCRN ────→ denoised ──┐
              └─ pVAD(raw) → gain ───────┤→ output = denoised × gain

設計重點：
  1. pVAD 直接從 raw audio 提取 speaker embedding（不經過 GTCRN）
  2. enrollment 使用 augmented embedding（多噪音條件的 centroid）
  3. pVAD 窗口加大到 1.5 秒（raw audio 品質較差，需要更多上下文）
  4. GTCRN 只負責降噪，不影響 pVAD 判斷
  5. 最終輸出 = denoised_audio × soft_gain

理由：
  串行版中 pVAD 依賴 GTCRN 的降噪品質；當 GTCRN 在高噪環境下失效
  （如 scenario C），pVAD 也跟著崩潰。並行版讓 pVAD 建立自己的噪音
  魯棒性（augmented enrollment + 更大窗口），與 SE 解耦。

用法:
    python pipeline_parallel.py -e enroll.wav -i mixed.wav -o output.wav
"""

import argparse
import time
import numpy as np
from pathlib import Path

from utils.audio import SAMPLE_RATE, read_audio, write_audio, frame_signal, overlap_add
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, cosine_similarity
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

# Fbank 最小 samples（與 speaker_encoder.py 一致）
FBANK_WIN_LENGTH = 400


class ParallelPVAD:
    """
    並行版 pVAD — 從 raw audio 提取 embedding + EMA 平滑 + 能量 VAD。

    設計要點：
    - 使用 CAM++（0.5s 窗口是穩定 embedding 的最低要求）
    - EMA 平滑 similarity（用歷史資訊補償短窗口的不穩定性）
    - 能量 VAD：靜音段不更新 similarity，同時主動衰減 EMA sim
    - 能量突變偵測：短時能量模式改變時加速 EMA 反應（提高 alpha）

    Parameters
    ----------
    speaker_encoder : SpeakerEncoder
    enrollment_dvector : np.ndarray
        enrollment embedding (embed_dim,)
    extract_interval : int
        每幾幀提取一次 embedding（預設 32 = ~0.5s at 16ms hop）
    window_sec : float
        提取 embedding 的滑窗長度（預設 0.5 秒）
    threshold : float
        cosine similarity 閾值
    ema_alpha : float
        EMA 平滑係數 (0~1)。越大表示新值權重越大，反應越快。
    energy_floor : float
        能量 VAD 閾值（RMS），低於此視為靜音
    energy_change_ratio : float
        能量突變比例閾值，超過時加速 EMA 反應
    """

    def __init__(
        self,
        speaker_encoder: SpeakerEncoder,
        enrollment_dvector: np.ndarray,
        extract_interval: int = 32,
        window_sec: float = 0.5,
        threshold: float = 0.25,
        ema_alpha: float = 0.5,
        energy_floor: float = 0.005,
        energy_change_ratio: float = 3.0,
    ):
        self.encoder = speaker_encoder
        self.enrollment = enrollment_dvector
        self.threshold = threshold
        self.extract_interval = extract_interval
        self.window_samples = int(window_sec * SAMPLE_RATE)
        self.ema_alpha = ema_alpha
        self.energy_floor = energy_floor
        self.energy_change_ratio = energy_change_ratio

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.frame_count = 0
        self._raw_sim = 0.0
        self._ema_sim = 0.0
        self._ema_initialized = False
        self._cached_is_target = False
        self._prev_rms = 0.0
        self._silence_frames = 0

    def reset(self) -> None:
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.frame_count = 0
        self._raw_sim = 0.0
        self._ema_sim = 0.0
        self._ema_initialized = False
        self._cached_is_target = False
        self._prev_rms = 0.0
        self._silence_frames = 0

    def _compute_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2) + 1e-12))

    def process_frame(self, raw_frame: np.ndarray) -> tuple:
        """
        處理一幀 raw audio，回傳 (is_target, ema_similarity)。

        - 能量 VAD：靜音段不提取 embedding，主動衰減 EMA sim
        - 能量突變：加速 EMA 反應（提高 alpha），讓新說話者的 sim 快速生效
        - similarity 經過 EMA 平滑

        Returns
        -------
        is_target : bool
        similarity : float (EMA smoothed)
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, raw_frame])
        if len(self.audio_buffer) > self.window_samples:
            self.audio_buffer = self.audio_buffer[-self.window_samples:]

        self.frame_count += 1

        # 幀級能量
        frame_rms = self._compute_rms(raw_frame)

        # 能量 VAD：靜音段主動衰減 EMA sim，避免前一人的 sim 殘留
        if frame_rms < self.energy_floor:
            self._silence_frames += 1
            # 靜音持續越久，EMA sim 衰減越快（但更溫和，避免短暫停頓誤衰減）
            decay = 0.98 ** self._silence_frames
            self._ema_sim *= decay
            self._cached_is_target = self._ema_sim > self.threshold
            self._prev_rms = frame_rms
            return self._cached_is_target, self._ema_sim

        self._silence_frames = 0

        # 能量突變偵測：加速 EMA 反應（不重置，但提高 alpha）
        effective_alpha = self.ema_alpha
        if self._prev_rms > self.energy_floor:
            ratio = frame_rms / (self._prev_rms + 1e-12)
            if ratio > self.energy_change_ratio or ratio < (1.0 / self.energy_change_ratio):
                # 能量突變 → 暫時提高 alpha 到 0.9，讓新 sim 快速主導
                effective_alpha = 0.9
        self._prev_rms = frame_rms

        should_extract = (self.frame_count % self.extract_interval == 0)

        if should_extract:
            min_samples = FBANK_WIN_LENGTH * 2
            if len(self.audio_buffer) >= min_samples:
                embedding = self.encoder.extract_embedding(self.audio_buffer)
                self._raw_sim = cosine_similarity(embedding, self.enrollment)

                # EMA 平滑（能量突變時用更高的 alpha 加速反應）
                if not self._ema_initialized:
                    self._ema_sim = self._raw_sim
                    self._ema_initialized = True
                else:
                    self._ema_sim = (effective_alpha * self._raw_sim
                                     + (1 - effective_alpha) * self._ema_sim)

                self._cached_is_target = self._ema_sim > self.threshold

        return self._cached_is_target, self._ema_sim


def run_parallel_pipeline(
    enrollment_path: str,
    input_path: str,
    output_path: str,
    threshold: float = 0.25,
    gain_floor: float = 0.0,
    attack_ms: float = 5.0,
    release_ms: float = 30.0,
    pvad_interval: int = 63,
    pvad_window_sec: float = 1.0,
    ema_alpha: float = 0.5,
    use_augmented_enrollment: bool = False,
    denoise_enrollment: bool = False,
):
    """
    並行版管線 v3：CAM++ + EMA + GTCRN 降噪。

    改進：
    - CAM++ 取代 WeSpeaker（0.5s 窗口區分力 +27%）
    - EMA α=0.5 平滑 similarity
    - 窗口 1.0s（最佳平衡：noisy 5dB 100% acc）
    - threshold 0.25（1.0s 窗口下 hsuan 100% / 0911 100% reject）

    Returns
    -------
    dict with keys:
        output_audio, denoised_audio, similarities, is_targets, n_frames
    """
    print("\n" + "=" * 60)
    print("並行版 pVAD + GTCRN SE 管線 (v3: CAM++ + EMA)")
    print(f"  pVAD model: CAM++, window: {pvad_window_sec}s")
    print(f"  threshold: {threshold}, EMA alpha: {ema_alpha}")
    print(f"  augmented enrollment: {use_augmented_enrollment}")
    print("=" * 60)

    # ── 1. 載入模型 ──────────────────────────────────
    print("\n[1/5] 載入模型...")
    campp_path = MODELS_DIR / "campplus" / "campplus.onnx"
    if not campp_path.exists():
        # fallback to WeSpeaker
        campp_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
        print("  [WARN] CAM++ not found, falling back to WeSpeaker")
    assert campp_path.exists(), f"找不到 speaker encoder ONNX: {campp_path}"

    speaker_encoder = SpeakerEncoder(str(campp_path))
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
    if denoise_enrollment:
        enrollment_audio = denoiser.enhance(enrollment_audio)
        print("  enrollment 已做 GTCRN 降噪")
    if use_augmented_enrollment:
        enrollment_dvector = speaker_encoder.extract_augmented_embedding(enrollment_audio)
        print(f"  使用 augmented enrollment（5 條件 centroid）")
    else:
        enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)
    print(f"  enrollment 長度: {len(enrollment_audio) / SAMPLE_RATE:.2f}s")
    print(f"  d-vector 維度: {enrollment_dvector.shape}")

    # ── 3. 載入混合音頻 ──────────────────────────────
    print("\n[3/5] 載入混合音頻...")
    input_audio = read_audio(input_path)
    print(f"  混合音頻長度: {len(input_audio) / SAMPLE_RATE:.2f}s")

    # ── 4. 並行處理 ──────────────────────────────────
    print("\n[4/5] 並行處理: GTCRN 降噪 ∥ pVAD(raw)...")
    t_start = time.time()

    # 路線 A：GTCRN 離線降噪（整段）
    denoised_audio = denoiser.enhance(input_audio)
    t_denoise = time.time() - t_start
    print(f"  GTCRN 降噪完成: {t_denoise:.2f}s")

    # 路線 B：pVAD 從 raw audio 提取 similarity（CAM++ + EMA）
    pvad = ParallelPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        window_sec=pvad_window_sec,
        threshold=threshold,
        ema_alpha=ema_alpha,
    )

    # 用 GTCRN hop 切幀（與降噪結果對齊）
    # 注意：pVAD 吃 raw audio，gating 套用在 denoised audio 上
    raw_frames_for_pvad = frame_signal(
        input_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP
    )
    denoised_frames = frame_signal(
        denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP
    )
    n_frames = min(raw_frames_for_pvad.shape[0], denoised_frames.shape[0])
    print(f"  pVAD 總幀數: {n_frames}, 每 {pvad_interval} 幀提取一次 embedding")

    enhanced_frames = np.empty((n_frames, GTCRN_NFFT), dtype=np.float32)
    similarities = []
    is_targets = []

    for i in range(n_frames):
        # pVAD 用 raw audio 的 hop-size samples
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(input_audio))
        raw_frame_samples = input_audio[start:end]

        is_target, sim = pvad.process_frame(raw_frame_samples)
        similarities.append(sim)
        is_targets.append(is_target)

        # 合併：denoised × soft_gain
        enhanced_frames[i] = gate.process(denoised_frames[i], is_target, confidence=sim)

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

    output_audio = output_audio[:len(input_audio)]
    write_audio(output_path, output_audio)
    print(f"  輸出: {output_path}")

    # ── 統計 ──────────────────────────────────────────
    sims = np.array(similarities)
    print(f"\npVAD 統計 (parallel, raw audio):")
    print(f"  similarity 平均: {sims.mean():.3f}")
    print(f"  similarity 標準差: {sims.std():.3f}")
    print(f"  目標活躍幀比例: {(sims > threshold).mean():.1%}")
    print(f"  embedding 提取次數: {n_frames // pvad_interval}")

    return {
        "output_audio": output_audio,
        "denoised_audio": denoised_audio[:len(input_audio)],
        "similarities": np.array(similarities),
        "is_targets": np.array(is_targets),
        "n_frames": n_frames,
    }


def run_parallel_pipeline_from_audio(
    enrollment_path,
    input_audio: np.ndarray,
    denoiser: GTCRNDenoiser,
    speaker_encoder: SpeakerEncoder,
    threshold: float = 0.25,
    gain_floor: float = 0.0,
    pvad_interval: int = 63,
    pvad_window_sec: float = 1.0,
    ema_alpha: float = 0.5,
    use_augmented_enrollment: bool = False,
    denoise_enrollment: bool = False,
    pvad_source: str = "raw",
):
    """
    給測試腳本呼叫的 API — 接受已載入的模型和 numpy 音頻。

    Parameters
    ----------
    pvad_source : "raw" | "denoised"
        "raw": pVAD 從原始音頻提取 embedding（並行架構）
        "denoised": pVAD 從 GTCRN 降噪後音頻提取（串行架構）

    Returns dict with output_audio, denoised_audio, similarities, is_targets, n_frames.
    """
    # Enrollment
    enrollment_audio = read_audio(str(enrollment_path))
    if denoise_enrollment:
        enrollment_audio = denoiser.enhance(enrollment_audio)
    if use_augmented_enrollment:
        enrollment_dvector = speaker_encoder.extract_augmented_embedding(enrollment_audio)
    else:
        enrollment_dvector = speaker_encoder.extract_embedding(enrollment_audio)

    # Route A: GTCRN denoise
    denoised_audio = denoiser.enhance(input_audio)

    # Route B: pVAD on raw audio (CAM++ + EMA)
    pvad = ParallelPVAD(
        speaker_encoder=speaker_encoder,
        enrollment_dvector=enrollment_dvector,
        extract_interval=pvad_interval,
        window_sec=pvad_window_sec,
        threshold=threshold,
        ema_alpha=ema_alpha,
    )

    gate = SoftGate(gain_floor=gain_floor, attack_ms=5.0, release_ms=30.0, hop=GTCRN_HOP)

    # pVAD 音源選擇
    pvad_audio = denoised_audio if pvad_source == "denoised" else input_audio

    denoised_frames = frame_signal(denoised_audio, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = denoised_frames.shape[0]

    enhanced_frames = np.empty((n_frames, GTCRN_NFFT), dtype=np.float32)
    similarities = []
    is_targets_list = []

    for i in range(n_frames):
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(pvad_audio))
        frame_samples = pvad_audio[start:end]

        is_target, sim = pvad.process_frame(frame_samples)
        similarities.append(sim)
        is_targets_list.append(is_target)

        enhanced_frames[i] = gate.process(denoised_frames[i], is_target, confidence=sim)

    output_audio = overlap_add(enhanced_frames, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    output_audio = output_audio[:len(input_audio)]
    peak = np.max(np.abs(output_audio))
    if peak > 0.99:
        output_audio = output_audio * 0.99 / peak

    return {
        "output_audio": output_audio,
        "denoised_audio": denoised_audio[:len(input_audio)],
        "similarities": np.array(similarities),
        "is_targets": np.array(is_targets_list),
        "n_frames": n_frames,
    }


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="並行版 pVAD + GTCRN SE 管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--enrollment", "-e", required=True,
                        help="enrollment 音頻路徑")
    parser.add_argument("--input", "-i", required=True,
                        help="混合音頻路徑")
    parser.add_argument("--output", "-o", default="output_parallel.wav",
                        help="輸出音頻路徑")
    parser.add_argument("--threshold", "-t", type=float, default=0.25,
                        help="pVAD 閾值（預設 0.25）")
    parser.add_argument("--gain-floor", type=float, default=0.05,
                        help="最小增益（預設 0.05）")
    parser.add_argument("--attack-ms", type=float, default=5.0,
                        help="attack 時間 ms")
    parser.add_argument("--release-ms", type=float, default=50.0,
                        help="release 時間 ms")
    parser.add_argument("--pvad-interval", type=int, default=63,
                        help="pVAD embedding 提取間隔（幀數，預設 63 ≈ 1.0s）")
    parser.add_argument("--pvad-window", type=float, default=1.0,
                        help="pVAD 滑窗長度 (秒，預設 1.0)")
    parser.add_argument("--ema-alpha", type=float, default=0.5,
                        help="EMA 平滑係數 (0~1，預設 0.5)")
    parser.add_argument("--augment", action="store_true",
                        help="使用 augmented enrollment（預設不使用）")

    args = parser.parse_args()

    run_parallel_pipeline(
        enrollment_path=args.enrollment,
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        gain_floor=args.gain_floor,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
        pvad_interval=args.pvad_interval,
        pvad_window_sec=args.pvad_window,
        ema_alpha=args.ema_alpha,
        use_augmented_enrollment=args.augment,
    )


if __name__ == "__main__":
    main()
