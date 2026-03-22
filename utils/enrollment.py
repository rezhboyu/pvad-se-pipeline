#!/usr/bin/env python3
"""
聲紋註冊模組 (Speaker Enrollment)
==================================
pVAD 的前置步驟：錄製或載入使用者語音 → 品質檢查 → 提取 d-vector → 存檔。

支援三種註冊方式：
1. 單段音檔註冊（最簡單，≥3 秒）
2. 多段合併註冊（3~5 段短句取 centroid，更穩定）
3. 增量更新（使用中逐步更新 d-vector）

用法:
    from utils.enrollment import SpeakerEnrollment

    enroll = SpeakerEnrollment()
    result = enroll.enroll_from_file("my_voice.wav")
    # result["dvector_path"] → "profiles/my_voice.npy"

    # 之後載入
    dvector = enroll.load_profile("my_voice")
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from .audio import SAMPLE_RATE, read_audio, write_audio
from .speaker_encoder import SpeakerEncoder, cosine_similarity


# ── 品質門檻 ──
MIN_DURATION_SEC = 3.0       # 最短錄音長度
MAX_DURATION_SEC = 30.0      # 最長錄音長度（避免記憶體爆炸）
MIN_SPEECH_RMS = 0.005       # 最小 RMS（太小 = 沒聲音或太遠）
MAX_SILENCE_RATIO = 0.7      # 靜音佔比上限（超過 = 大部分是安靜的）
MIN_SNR_DB = 5.0             # 最低信噪比
SILENCE_THRESHOLD = 0.01     # 靜音判定閾值（RMS）

# ── 註冊設定 ──
MULTI_SEGMENT_MIN = 2        # 多段註冊最少段數
MULTI_SEGMENT_MAX = 5        # 多段註冊最多段數
CONSISTENCY_THRESHOLD = 0.5  # 多段之間的最低相似度（確保是同一人）


@dataclass
class QualityReport:
    """音頻品質報告。"""
    duration_sec: float
    rms: float
    peak: float
    snr_db: float
    silence_ratio: float
    is_valid: bool
    issues: list


@dataclass
class EnrollmentResult:
    """註冊結果。"""
    success: bool
    profile_name: str
    dvector_path: str
    embed_dim: int
    n_segments: int
    quality: dict
    timestamp: str
    message: str


class SpeakerEnrollment:
    """
    聲紋註冊器。

    Parameters
    ----------
    encoder : SpeakerEncoder
        已載入的 speaker encoder（CAM++ 推薦）
    profiles_dir : str or Path
        存放聲紋檔的目錄（預設 profiles/）
    """

    def __init__(
        self,
        encoder: SpeakerEncoder,
        profiles_dir: str | Path = "profiles",
    ):
        self.encoder = encoder
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # 品質檢查
    # ═══════════════════════════════════════════════════════
    def check_quality(self, audio: np.ndarray) -> QualityReport:
        """
        檢查音頻品質，回傳報告。

        檢查項目：
        1. 長度 ≥ 3 秒
        2. RMS ≥ 0.005（有足夠音量）
        3. 靜音佔比 < 70%（有足夠語音內容）
        4. SNR ≥ 5 dB（不太吵）
        """
        issues = []

        # 長度
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION_SEC:
            issues.append(f"太短: {duration:.1f}s < {MIN_DURATION_SEC}s")
        if duration > MAX_DURATION_SEC:
            issues.append(f"太長: {duration:.1f}s > {MAX_DURATION_SEC}s（建議裁剪）")

        # RMS（整體音量）
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        peak = float(np.max(np.abs(audio)))
        if rms < MIN_SPEECH_RMS:
            issues.append(f"音量太低: RMS={rms:.4f} < {MIN_SPEECH_RMS}")

        # 靜音佔比
        frame_len = int(0.025 * SAMPLE_RATE)  # 25ms frames
        hop = int(0.010 * SAMPLE_RATE)         # 10ms hop
        n_frames = max(1, (len(audio) - frame_len) // hop + 1)
        silence_count = 0
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + frame_len]
            frame_rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            if frame_rms < SILENCE_THRESHOLD:
                silence_count += 1
        silence_ratio = silence_count / n_frames

        if silence_ratio > MAX_SILENCE_RATIO:
            issues.append(f"靜音太多: {silence_ratio:.0%} > {MAX_SILENCE_RATIO:.0%}")

        # 簡易 SNR 估算（語音段 RMS / 靜音段 RMS）
        frame_rms_list = []
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + frame_len]
            frame_rms_list.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
        frame_rms_arr = np.array(frame_rms_list)

        # 將 frames 按 RMS 排序，前 20% 視為靜音，後 20% 視為語音
        sorted_rms = np.sort(frame_rms_arr)
        n20 = max(1, len(sorted_rms) // 5)
        noise_rms = np.mean(sorted_rms[:n20])
        speech_rms = np.mean(sorted_rms[-n20:])
        snr_db = float(20 * np.log10(speech_rms / (noise_rms + 1e-12)))

        if snr_db < MIN_SNR_DB:
            issues.append(f"SNR 太低: {snr_db:.1f}dB < {MIN_SNR_DB}dB（環境太吵）")

        is_valid = len(issues) == 0

        return QualityReport(
            duration_sec=round(duration, 2),
            rms=round(rms, 4),
            peak=round(peak, 4),
            snr_db=round(snr_db, 1),
            silence_ratio=round(silence_ratio, 3),
            is_valid=is_valid,
            issues=issues,
        )

    # ═══════════════════════════════════════════════════════
    # 單段註冊
    # ═══════════════════════════════════════════════════════
    def enroll_from_file(
        self,
        audio_path: str,
        profile_name: Optional[str] = None,
        skip_quality_check: bool = False,
    ) -> EnrollmentResult:
        """
        從單一音檔註冊聲紋。

        Parameters
        ----------
        audio_path : str
            音檔路徑（wav, 16kHz mono）
        profile_name : str, optional
            聲紋名稱。預設用檔名。
        skip_quality_check : bool
            跳過品質檢查（除錯用）

        Returns
        -------
        EnrollmentResult
        """
        audio_path = Path(audio_path)
        if profile_name is None:
            profile_name = audio_path.stem

        # 讀取音頻
        audio = read_audio(str(audio_path))

        return self.enroll_from_audio(
            audio, profile_name, skip_quality_check=skip_quality_check
        )

    def enroll_from_audio(
        self,
        audio: np.ndarray,
        profile_name: str,
        skip_quality_check: bool = False,
    ) -> EnrollmentResult:
        """
        從 numpy 音頻註冊聲紋。

        Parameters
        ----------
        audio : np.ndarray
            (n_samples,) float32, 16kHz mono
        profile_name : str
            聲紋名稱
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 品質檢查
        quality = self.check_quality(audio)
        if not skip_quality_check and not quality.is_valid:
            return EnrollmentResult(
                success=False,
                profile_name=profile_name,
                dvector_path="",
                embed_dim=0,
                n_segments=0,
                quality=asdict(quality),
                timestamp=timestamp,
                message=f"品質檢查未通過: {'; '.join(quality.issues)}",
            )

        # 裁剪到最大長度
        max_samples = int(MAX_DURATION_SEC * SAMPLE_RATE)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # 提取 d-vector
        dvector = self.encoder.extract_embedding(audio)

        # 存檔
        dvector_path = self.profiles_dir / f"{profile_name}.npy"
        np.save(str(dvector_path), dvector)

        # 存 metadata
        meta = {
            "profile_name": profile_name,
            "embed_dim": int(dvector.shape[0]),
            "n_segments": 1,
            "quality": asdict(quality),
            "timestamp": timestamp,
            "encoder_model": str(self.encoder.onnx_path.name),
        }
        meta_path = self.profiles_dir / f"{profile_name}.json"
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return EnrollmentResult(
            success=True,
            profile_name=profile_name,
            dvector_path=str(dvector_path),
            embed_dim=int(dvector.shape[0]),
            n_segments=1,
            quality=asdict(quality),
            timestamp=timestamp,
            message=f"註冊成功: {profile_name} (dim={dvector.shape[0]})",
        )

    # ═══════════════════════════════════════════════════════
    # 多段合併註冊（更穩定）
    # ═══════════════════════════════════════════════════════
    def enroll_multi_segment(
        self,
        audio_paths: list[str],
        profile_name: str,
        skip_quality_check: bool = False,
    ) -> EnrollmentResult:
        """
        從多段音檔註冊聲紋（取 centroid）。

        多段的好處：
        - 涵蓋不同語速、音調 → 更泛化的 d-vector
        - 自動檢查段間一致性（確保是同一人）

        Parameters
        ----------
        audio_paths : list of str
            2~5 個音檔路徑
        profile_name : str
            聲紋名稱
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        n = len(audio_paths)
        if n < MULTI_SEGMENT_MIN:
            return EnrollmentResult(
                success=False, profile_name=profile_name, dvector_path="",
                embed_dim=0, n_segments=n, quality={},
                timestamp=timestamp,
                message=f"段數不足: {n} < {MULTI_SEGMENT_MIN}",
            )
        if n > MULTI_SEGMENT_MAX:
            audio_paths = audio_paths[:MULTI_SEGMENT_MAX]
            n = MULTI_SEGMENT_MAX

        # 逐段提取 embedding + 品質檢查
        embeddings = []
        qualities = []
        for path in audio_paths:
            audio = read_audio(str(path))
            quality = self.check_quality(audio)
            qualities.append(asdict(quality))

            if not skip_quality_check and not quality.is_valid:
                return EnrollmentResult(
                    success=False, profile_name=profile_name, dvector_path="",
                    embed_dim=0, n_segments=n,
                    quality={"segment_qualities": qualities},
                    timestamp=timestamp,
                    message=f"段 {len(embeddings)+1} 品質未通過: {'; '.join(quality.issues)}",
                )

            # 裁剪
            max_samples = int(MAX_DURATION_SEC * SAMPLE_RATE)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            emb = self.encoder.extract_embedding(audio)
            embeddings.append(emb)

        # 一致性檢查：所有段之間的 cosine similarity 都要 > threshold
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim < CONSISTENCY_THRESHOLD:
                    return EnrollmentResult(
                        success=False, profile_name=profile_name, dvector_path="",
                        embed_dim=int(embeddings[0].shape[0]), n_segments=n,
                        quality={"segment_qualities": qualities},
                        timestamp=timestamp,
                        message=f"段 {i+1} 和段 {j+1} 不一致 (sim={sim:.3f} < {CONSISTENCY_THRESHOLD})，"
                                f"可能不是同一人",
                    )

        # 計算 centroid
        centroid = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm

        # 存檔
        dvector_path = self.profiles_dir / f"{profile_name}.npy"
        np.save(str(dvector_path), centroid)

        # 計算段間統計
        inter_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                inter_sims.append(float(cosine_similarity(embeddings[i], embeddings[j])))

        meta = {
            "profile_name": profile_name,
            "embed_dim": int(centroid.shape[0]),
            "n_segments": n,
            "inter_segment_similarities": inter_sims,
            "mean_inter_sim": round(float(np.mean(inter_sims)), 4),
            "segment_qualities": qualities,
            "timestamp": timestamp,
            "encoder_model": str(self.encoder.onnx_path.name),
        }
        meta_path = self.profiles_dir / f"{profile_name}.json"
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return EnrollmentResult(
            success=True,
            profile_name=profile_name,
            dvector_path=str(dvector_path),
            embed_dim=int(centroid.shape[0]),
            n_segments=n,
            quality={"segment_qualities": qualities, "mean_inter_sim": float(np.mean(inter_sims))},
            timestamp=timestamp,
            message=f"多段註冊成功: {profile_name} ({n} 段, centroid dim={centroid.shape[0]}, "
                    f"段間 sim={np.mean(inter_sims):.3f})",
        )

    # ═══════════════════════════════════════════════════════
    # 增量更新
    # ═══════════════════════════════════════════════════════
    def update_profile(
        self,
        profile_name: str,
        new_audio: np.ndarray,
        update_weight: float = 0.1,
    ) -> EnrollmentResult:
        """
        增量更新現有聲紋（EMA 式）。

        使用場景：
        - 使用過程中，當確認是目標說話人時，用當前音頻微調 d-vector
        - 適應環境變化（不同麥克風、不同房間）

        Parameters
        ----------
        profile_name : str
            已註冊的聲紋名稱
        new_audio : np.ndarray
            新的音頻片段
        update_weight : float
            更新權重（0.1 = 10% 新值，90% 舊值）
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 載入現有 d-vector
        dvector_path = self.profiles_dir / f"{profile_name}.npy"
        if not dvector_path.exists():
            return EnrollmentResult(
                success=False, profile_name=profile_name, dvector_path="",
                embed_dim=0, n_segments=0, quality={},
                timestamp=timestamp,
                message=f"聲紋不存在: {profile_name}",
            )

        old_dvector = np.load(str(dvector_path))

        # 提取新 embedding
        new_emb = self.encoder.extract_embedding(new_audio)

        # 先驗證新 embedding 和舊的夠相似（防止誤更新）
        sim = cosine_similarity(new_emb, old_dvector)
        if sim < CONSISTENCY_THRESHOLD:
            return EnrollmentResult(
                success=False, profile_name=profile_name,
                dvector_path=str(dvector_path),
                embed_dim=int(old_dvector.shape[0]),
                n_segments=0, quality={},
                timestamp=timestamp,
                message=f"新音頻與現有聲紋不一致 (sim={sim:.3f} < {CONSISTENCY_THRESHOLD})，拒絕更新",
            )

        # EMA 更新
        updated = (update_weight * new_emb + (1 - update_weight) * old_dvector).astype(np.float32)
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            updated = updated / norm

        # 存檔（覆蓋）
        np.save(str(dvector_path), updated)

        # 更新 metadata
        meta_path = self.profiles_dir / f"{profile_name}.json"
        meta = {}
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                meta = json.load(f)

        update_count = meta.get("update_count", 0) + 1
        meta["last_updated"] = timestamp
        meta["update_count"] = update_count
        meta["last_update_sim"] = round(float(sim), 4)

        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return EnrollmentResult(
            success=True,
            profile_name=profile_name,
            dvector_path=str(dvector_path),
            embed_dim=int(updated.shape[0]),
            n_segments=update_count,
            quality={"update_sim": float(sim)},
            timestamp=timestamp,
            message=f"增量更新成功: {profile_name} (sim={sim:.3f}, 累計更新 {update_count} 次)",
        )

    # ═══════════════════════════════════════════════════════
    # 載入 / 管理
    # ═══════════════════════════════════════════════════════
    def load_profile(self, profile_name: str) -> np.ndarray:
        """載入已註冊的 d-vector。"""
        dvector_path = self.profiles_dir / f"{profile_name}.npy"
        if not dvector_path.exists():
            raise FileNotFoundError(f"聲紋不存在: {dvector_path}")
        return np.load(str(dvector_path))

    def list_profiles(self) -> list[dict]:
        """列出所有已註冊的聲紋。"""
        profiles = []
        for npy in sorted(self.profiles_dir.glob("*.npy")):
            name = npy.stem
            meta_path = self.profiles_dir / f"{name}.json"
            meta = {}
            if meta_path.exists():
                with open(str(meta_path), "r", encoding="utf-8") as f:
                    meta = json.load(f)
            profiles.append({
                "name": name,
                "dvector_path": str(npy),
                "embed_dim": meta.get("embed_dim", "?"),
                "n_segments": meta.get("n_segments", "?"),
                "timestamp": meta.get("timestamp", "?"),
                "encoder_model": meta.get("encoder_model", "?"),
            })
        return profiles

    def delete_profile(self, profile_name: str) -> bool:
        """刪除聲紋。"""
        dvector_path = self.profiles_dir / f"{profile_name}.npy"
        meta_path = self.profiles_dir / f"{profile_name}.json"
        deleted = False
        if dvector_path.exists():
            dvector_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
        return deleted

    def verify_speaker(
        self,
        profile_name: str,
        audio: np.ndarray,
        threshold: float = 0.25,
    ) -> tuple[bool, float]:
        """
        驗證音頻是否為已註冊的說話人。

        注意 threshold 的選擇：
        - 整段音頻驗證（≥3s）：threshold=0.25（預設）
        - 1.0s 窗口實時 pVAD：threshold=0.25（pipeline 中設定）
        - 噪音環境可適當降低

        Returns
        -------
        (is_match, similarity)
        """
        dvector = self.load_profile(profile_name)
        emb = self.encoder.extract_embedding(audio)
        sim = cosine_similarity(emb, dvector)
        is_match = sim > threshold
        return is_match, float(sim)
