#!/usr/bin/env python3
"""
Baseline Comparison: 舊架構 (VAD+SpeakerID) vs pVAD-SE Pipeline
================================================================
舊架構: Silero VAD (逐幀語音偵測) + CAM++ (整段 Speaker Verification)
        → VAD 偵測語音段 → Speaker ID 判斷是否目標 → 全放行或全靜音

新架構: CAM++ pVAD (逐幀目標說話者辨識) + GTCRN SE (降噪) + Gating
        → 逐幀判定是否目標說話者 → 降噪 + soft gating

比較方式:
  用我們的 CAM++ ONNX 模型模擬舊架構的兩階段流程，
  同時加入 SDMFCC SVM/MLP 作為額外 baseline。
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

import librosa

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

VOICE_DIR = Path.home() / "Desktop" / "VOICE"
SDMFCC_DIR = VOICE_DIR / "SDmfcc_optimized"
MAT_DIR = VOICE_DIR / "MAT"

from utils.audio import read_audio, SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder, cosine_similarity

MODELS_DIR = PROJECT_DIR / "models"

# ══════════════════════════════════════════════════════
# 1. 載入模型
# ══════════════════════════════════════════════════════

def load_svm_model():
    with open(SDMFCC_DIR / "results_ml" / "svm_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


def load_mlp_model():
    import torch
    sys.path.insert(0, str(SDMFCC_DIR))
    from v1_basic.models.model import create_model
    model = create_model({
        "model_type": "improved_mlp",
        "input_dim": 45,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "dropout_rate": 0.3,
        "use_residual": True,
    })
    checkpoint = torch.load(
        SDMFCC_DIR / "v1_basic" / "results" / "model_best.pth",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_sdmfcc(audio, sr=16000, hop_length=512):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T  # (T, 45)


def predict_svm(audio, model, scaler):
    features = extract_sdmfcc(audio)
    return model.predict(scaler.transform(features))


def predict_mlp(audio, model):
    import torch
    features = extract_sdmfcc(audio)
    with torch.no_grad():
        out = model(torch.FloatTensor(features).unsqueeze(0))
        return out.argmax(dim=-1).squeeze().numpy()


# ══════════════════════════════════════════════════════
# 2. 模擬舊架構: VAD + Speaker ID
# ══════════════════════════════════════════════════════

class OldPipeline:
    """
    模擬舊架構: Silero VAD + CAM++ Speaker Verification

    流程:
    1. 用 SDMFCC SVM 做逐幀 VAD（模擬 Silero VAD 的角色）
    2. 對每個語音段提取 CAM++ embedding
    3. 與 enrollment 比較 cosine similarity
    4. 超過閾值 → 該段標記為目標；否則 → 全靜音

    與舊架構的差異:
    - 用 SVM VAD 代替 Silero VAD（Silero 需要 torch.hub）
    - CAM++ 用 ONNX 而非 ModelScope
    - 判定粒度: 逐語音段（而非整段音訊）
    """

    def __init__(self, speaker_encoder, enrollment_dvector,
                 svm_model, svm_scaler, speaker_threshold=0.25,
                 segment_mode="per_segment"):
        self.encoder = speaker_encoder
        self.enrollment = enrollment_dvector
        self.svm = svm_model
        self.scaler = svm_scaler
        self.speaker_threshold = speaker_threshold
        self.segment_mode = segment_mode  # "per_segment" or "whole_audio"

    def process(self, audio):
        """
        Returns per-frame is_target decisions at SDMFCC frame rate (32ms).
        """
        # Step 1: VAD (SDMFCC SVM, per-frame)
        vad_preds = predict_svm(audio, self.svm, self.scaler)  # 0/1
        n_frames = len(vad_preds)
        hop = 512
        is_target = np.zeros(n_frames, dtype=bool)

        if self.segment_mode == "whole_audio":
            # 舊架構的原始做法: 對整段音訊做一次 Speaker ID
            emb = self.encoder.extract_embedding(audio)
            sim = cosine_similarity(emb, self.enrollment)
            if sim >= self.speaker_threshold:
                is_target = vad_preds.astype(bool)  # VAD=1 的幀都是目標
            # else: 全部靜音
            return is_target, sim

        else:
            # 改良版: 對每個語音段分別做 Speaker ID
            segments = self._get_speech_segments(vad_preds, hop)
            overall_sim = 0.0

            for seg_start, seg_end in segments:
                seg_audio = audio[seg_start:seg_end]
                if len(seg_audio) < SAMPLE_RATE * 0.3:  # 太短跳過
                    continue
                emb = self.encoder.extract_embedding(seg_audio)
                sim = cosine_similarity(emb, self.enrollment)
                overall_sim = max(overall_sim, sim)

                if sim >= self.speaker_threshold:
                    frame_start = seg_start // hop
                    frame_end = min(seg_end // hop, n_frames)
                    is_target[frame_start:frame_end] = True

            return is_target, overall_sim

    def _get_speech_segments(self, vad_preds, hop, min_dur_frames=8):
        """將連續的 VAD=1 幀合併成語音段"""
        segments = []
        in_speech = False
        start = 0
        for i, v in enumerate(vad_preds):
            if v == 1 and not in_speech:
                start = i
                in_speech = True
            elif v == 0 and in_speech:
                if i - start >= min_dur_frames:
                    segments.append((start * hop, i * hop))
                in_speech = False
        if in_speech and len(vad_preds) - start >= min_dur_frames:
            segments.append((start * hop, len(vad_preds) * hop))
        return segments


# ══════════════════════════════════════════════════════
# 3. 建立測試場景
# ══════════════════════════════════════════════════════

def build_scenarios():
    hsuan_files = sorted(MAT_DIR.glob("*hsuan_7.wav"))
    interf_files = sorted(MAT_DIR.glob("*0911*_7.wav"))
    femh_files = sorted(MAT_DIR.glob("*FEMH_7.wav"))

    hsuan_audios = [read_audio(str(f)) for f in hsuan_files]
    interf_audios = [read_audio(str(f)) for f in interf_files]
    femh_audio = read_audio(str(femh_files[0])) if femh_files else None

    seg_len = int(2.5 * SAMPLE_RATE)

    def get_seg(audios, idx, start=0):
        a = audios[idx % len(audios)]
        return a[start:start + seg_len]

    # A: hsuan ↔ 0911
    scenario_a = np.concatenate([
        get_seg(hsuan_audios, 0), get_seg(interf_audios, 0),
        get_seg(hsuan_audios, 1), get_seg(interf_audios, 1),
        get_seg(hsuan_audios, 2),
    ])
    a_labels = [
        ("hsuan_0.0-2.5s", 0.0, 2.5, True),
        ("interferer_2.5-5.0s", 2.5, 5.0, False),
        ("hsuan_5.0-7.5s", 5.0, 7.5, True),
        ("interferer_7.5-10.0s", 7.5, 10.0, False),
        ("hsuan_10.0-12.5s", 10.0, 12.5, True),
    ]

    # B: overlap
    hsuan_b = get_seg(hsuan_audios, 0)
    overlap_len = int(3.0 * SAMPLE_RATE)
    overlap = hsuan_audios[0][seg_len:seg_len + overlap_len] + interf_audios[0][:overlap_len]
    scenario_b = np.concatenate([hsuan_b, overlap, get_seg(hsuan_audios, 1)])
    b_labels = [
        ("hsuan_only_0.0-2.5s", 0.0, 2.5, True),
        ("hsuan+interferer_2.5-5.5s", 2.5, 5.5, True),
        ("hsuan_only_5.5-8.0s", 5.5, 8.0, True),
    ]

    # C: noisy
    noise = np.random.RandomState(42).randn(len(scenario_a)).astype(np.float32)
    sig_power = np.mean(scenario_a ** 2)
    noise_power = sig_power / (10 ** (10 / 10))
    noise *= np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-12))
    scenario_c = scenario_a + noise
    c_labels = a_labels.copy()

    # D: hsuan ↔ FEMH
    if femh_audio is not None:
        scenario_d = np.concatenate([
            get_seg(hsuan_audios, 0), femh_audio[:seg_len],
            get_seg(hsuan_audios, 1), femh_audio[seg_len:2 * seg_len],
            get_seg(hsuan_audios, 2),
        ])
    else:
        scenario_d = scenario_a.copy()
    d_labels = [
        ("hsuan_0.0-2.5s", 0.0, 2.5, True),
        ("interferer_2.5-5.0s", 2.5, 5.0, False),
        ("hsuan_5.0-7.5s", 5.0, 7.5, True),
        ("interferer_7.5-10.0s", 7.5, 10.0, False),
        ("hsuan_10.0-12.5s", 10.0, 12.5, True),
    ]

    return {
        "scenario_a": (scenario_a, a_labels),
        "scenario_b": (scenario_b, b_labels),
        "scenario_c": (scenario_c, c_labels),
        "scenario_d": (scenario_d, d_labels),
    }


# ══════════════════════════════════════════════════════
# 4. 評估
# ══════════════════════════════════════════════════════

def evaluate(predictions, labels, hop=512):
    """計算每段的 target_ratio（判定為目標的比例）"""
    results = {}
    frame_dur = hop / SAMPLE_RATE

    for name, start_sec, end_sec, is_target in labels:
        sf = int(start_sec / frame_dur)
        ef = min(int(end_sec / frame_dur), len(predictions))
        seg = predictions[sf:ef]
        if len(seg) == 0:
            continue
        results[name] = {
            "target_ratio": round(float(np.mean(seg)), 4),
            "is_target": is_target,
            "n_frames": len(seg),
        }
    return results


def main():
    print("=" * 70)
    print("Baseline Comparison: 舊架構 vs pVAD-SE Pipeline")
    print("=" * 70)

    # 載入模型
    print("\n[1/5] 載入模型...")
    svm_model, svm_scaler = load_svm_model()
    mlp_model = load_mlp_model()
    speaker_encoder = SpeakerEncoder(str(MODELS_DIR / "campplus" / "campplus.onnx"))
    print("  SVM, MLP, CAM++ loaded")

    # Enrollment
    print("\n[2/5] 提取 enrollment d-vector...")
    enroll_path = str(MAT_DIR / "b6dbc0fc-1d57-4647-aa85-54f9bea08743_hsuan_7.wav")
    enroll_audio = read_audio(enroll_path)
    enrollment_dvector = speaker_encoder.extract_embedding(enroll_audio)
    print(f"  enrollment: {len(enroll_audio)/SAMPLE_RATE:.1f}s, SNR=44dB")

    # 建立場景
    print("\n[3/5] 建立測試場景...")
    scenarios = build_scenarios()

    # 建立舊架構 pipeline
    old_whole = OldPipeline(speaker_encoder, enrollment_dvector,
                            svm_model, svm_scaler,
                            speaker_threshold=0.25, segment_mode="whole_audio")
    old_seg = OldPipeline(speaker_encoder, enrollment_dvector,
                          svm_model, svm_scaler,
                          speaker_threshold=0.25, segment_mode="per_segment")

    # 跑測試
    print("\n[4/5] 執行測試...")
    all_results = {}

    for sc_name, (audio, labels) in scenarios.items():
        print(f"\n  --- {sc_name} ({len(audio)/SAMPLE_RATE:.1f}s) ---")
        all_results[sc_name] = {}

        # A) SDMFCC SVM (VAD only)
        svm_preds = predict_svm(audio, svm_model, svm_scaler)
        all_results[sc_name]["svm_vad_only"] = evaluate(svm_preds, labels)

        # B) SDMFCC MLP (VAD only)
        mlp_preds = predict_mlp(audio, mlp_model)
        all_results[sc_name]["mlp_vad_only"] = evaluate(mlp_preds, labels)

        # C) 舊架構: SVM VAD + CAM++ Speaker ID (整段)
        old_w_preds, old_w_sim = old_whole.process(audio)
        all_results[sc_name]["old_whole"] = evaluate(old_w_preds.astype(int), labels)
        all_results[sc_name]["old_whole"]["speaker_sim"] = round(old_w_sim, 4)

        # D) 舊架構改良: SVM VAD + CAM++ Speaker ID (逐段)
        old_s_preds, old_s_sim = old_seg.process(audio)
        all_results[sc_name]["old_per_seg"] = evaluate(old_s_preds.astype(int), labels)
        all_results[sc_name]["old_per_seg"]["speaker_sim"] = round(old_s_sim, 4)

        # 打印
        for seg_name in all_results[sc_name]["svm_vad_only"]:
            gt = "T" if all_results[sc_name]["svm_vad_only"][seg_name]["is_target"] else "I"
            svm_tr = all_results[sc_name]["svm_vad_only"][seg_name]["target_ratio"]
            mlp_tr = all_results[sc_name]["mlp_vad_only"][seg_name]["target_ratio"]
            old_w_tr = all_results[sc_name]["old_whole"].get(seg_name, {}).get("target_ratio", 0)
            old_s_tr = all_results[sc_name]["old_per_seg"].get(seg_name, {}).get("target_ratio", 0)
            print(f"    {seg_name:30s} [{gt}] SVM={svm_tr:.1%} MLP={mlp_tr:.1%} "
                  f"Old_W={old_w_tr:.1%} Old_S={old_s_tr:.1%}")

    # 載入 pVAD-SE 結果
    print("\n[5/5] 比較報告...")
    pvad_report = PROJECT_DIR / "test_parallel" / "test_report_parallel.json"
    pvad_metrics = {}
    if pvad_report.exists():
        with open(pvad_report) as f:
            pvad_metrics = json.load(f).get("metrics", {})

    # ══════════════════════════════════════════════════
    # 比較表
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    print()
    print("模型說明:")
    print("  SVM_VAD    : SDMFCC SVM, 只做 VAD（有/無語音），不辨識說話者")
    print("  MLP_VAD    : SDMFCC MLP, 同上")
    print("  OLD_WHOLE  : SVM VAD + CAM++ 整段 Speaker ID（舊架構）")
    print("  OLD_PERSEG : SVM VAD + CAM++ 逐語音段 Speaker ID（改良舊架構）")
    print("  pVAD-SE    : CAM++ 逐幀 pVAD + GTCRN SE + Energy VAD + Gating（我們的）")
    print()

    header = f"{'Segment':32s} {'SVM_VAD':>8s} {'MLP_VAD':>8s} {'OLD_W':>8s} {'OLD_S':>8s} {'pVAD-SE':>8s} {'GT':>6s}"
    print(header)
    print("-" * len(header))

    for sc_name in ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]:
        print(f"\n  [{sc_name.upper()}]")
        sc = all_results.get(sc_name, {})
        pvad_sc = pvad_metrics.get(sc_name, {})

        for seg_name in sc.get("svm_vad_only", {}):
            gt = "T" if sc["svm_vad_only"][seg_name]["is_target"] else "I"
            svm = sc["svm_vad_only"][seg_name]["target_ratio"]
            mlp = sc["mlp_vad_only"].get(seg_name, {}).get("target_ratio", 0)
            old_w = sc["old_whole"].get(seg_name, {}).get("target_ratio", 0)
            old_s = sc["old_per_seg"].get(seg_name, {}).get("target_ratio", 0)
            pvad = pvad_sc.get(seg_name, {}).get("target_ratio", -1)
            pvad_s = f"{pvad:.1%}" if pvad >= 0 else "N/A"

            print(f"  {seg_name:32s} {svm:7.1%} {mlp:8.1%} {old_w:7.1%} {old_s:7.1%} {pvad_s:>8s} {gt:>6s}")

    # 存檔
    output = {
        "description": "舊架構 (VAD+SpeakerID) vs pVAD-SE Pipeline 比較",
        "models": {
            "svm_vad_only": "SDMFCC 45-dim + SVM RBF (VAD only)",
            "mlp_vad_only": "SDMFCC 45-dim + MLP 256-128-64 (VAD only)",
            "old_whole": "SVM VAD + CAM++ 整段 Speaker ID (舊架構)",
            "old_per_seg": "SVM VAD + CAM++ 逐語音段 Speaker ID (改良舊架構)",
            "pvad_se": "CAM++ pVAD + GTCRN SE + Energy VAD + Gating (新架構)",
        },
        "enrollment": "b6dbc0fc_hsuan_7.wav (SNR=44dB, 19.9s)",
        "results": all_results,
    }
    out_path = PROJECT_DIR / "test_parallel" / "baseline_comparison_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\n報告已存: {out_path}")


if __name__ == "__main__":
    main()
