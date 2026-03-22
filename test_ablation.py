#!/usr/bin/env python3
"""
消融實驗 — pVAD-SE Pipeline v3
================================
逐一變更關鍵參數，量化各元件對效能的貢獻度。

Baseline:
  enrollment: b6dbc0fc_hsuan_7.wav (SNR 44dB)
  augmented: False, threshold: 0.25, window: 0.5s
  ema_alpha: 0.6, gain_floor: 0.0, release_ms: 30

輸出:
  test_ablation/ablation_report.json  — 完整數據
  test_ablation/ablation_summary.csv  — 匯總表
  test_ablation/ablation_heatmap.png  — 視覺化

用法:
    python test_ablation.py
"""

import sys
import os
import json
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE, read_audio, write_audio, frame_signal, overlap_add
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP
from pipeline_parallel import run_parallel_pipeline_from_audio

# ── 路徑 ──────────────────────────────────────────────
MAT_DIR = Path(os.path.expanduser("~/Desktop/VOICE/MAT"))
assert MAT_DIR.exists(), f"找不到 MAT 目錄: {MAT_DIR}"

OUTPUT_DIR = PROJECT_DIR / "test_ablation"
OUTPUT_DIR.mkdir(exist_ok=True)

# 音檔
ENROLLMENT_CLEAN  = MAT_DIR / "b6dbc0fc-1d57-4647-aa85-54f9bea08743_hsuan_7.wav"   # SNR 44dB
ENROLLMENT_NOISY  = MAT_DIR / "275eaceb-2387-4f9e-aef5-1e9996b8f024_hsuan_7.wav"   # SNR 10dB
HSUAN_SOURCE      = MAT_DIR / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"
INTERFERER_0911   = MAT_DIR / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"
INTERFERER_FEMH   = MAT_DIR / "ef81890f-791f-45c5-925a-2e7931a8f3c6_FEMH_7.wav"


# ── 工具函式（從 test_parallel_pipeline.py 複用）──────
def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)

def mix_at_snr(signal, noise, snr_db):
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return signal + noise * scale

def trim_or_pad(audio, length):
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])


# ── 場景建構 ──────────────────────────────────────────
def build_scenario_a(hsuan_audio, interferer_audio, seg_sec=2.5):
    seg_len = int(seg_sec * SAMPLE_RATE)
    segments, labels = [], []
    t, h_off, i_off = 0.0, 0, 0
    for idx in range(5):
        if idx % 2 == 0:
            seg = trim_or_pad(hsuan_audio[h_off:h_off + seg_len], seg_len)
            segments.append(seg)
            labels.append((t, t + seg_sec, "hsuan"))
            h_off += seg_len
        else:
            seg = trim_or_pad(interferer_audio[i_off:i_off + seg_len], seg_len)
            segments.append(seg)
            labels.append((t, t + seg_sec, "interferer"))
            i_off += seg_len
        t += seg_sec
    return np.concatenate(segments), labels

def build_scenario_b(hsuan_audio, interferer_audio, total_sec=8.0,
                     overlap_start_sec=2.5, overlap_end_sec=5.5):
    total_len = int(total_sec * SAMPLE_RATE)
    hsuan_seg = trim_or_pad(hsuan_audio, total_len)
    ov_s = int(overlap_start_sec * SAMPLE_RATE)
    ov_e = int(overlap_end_sec * SAMPLE_RATE)
    inter_seg = trim_or_pad(interferer_audio, ov_e - ov_s)
    scale = rms(hsuan_seg[ov_s:ov_e]) / (rms(inter_seg) + 1e-12)
    mixed = hsuan_seg.copy()
    mixed[ov_s:ov_e] += inter_seg * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * 0.95 / peak
    labels = [
        (0.0, overlap_start_sec, "hsuan_only"),
        (overlap_start_sec, overlap_end_sec, "hsuan+interferer"),
        (overlap_end_sec, total_sec, "hsuan_only"),
    ]
    return mixed, labels

def build_scenario_c(hsuan_audio, interferer_audio, seg_sec=2.5, noise_snr_db=10):
    mixed_clean, labels = build_scenario_a(hsuan_audio, interferer_audio, seg_sec=seg_sec)
    np.random.seed(42)  # 固定隨機種子確保可重複性
    noise = np.random.randn(len(mixed_clean)).astype(np.float32)
    mixed_noisy = mix_at_snr(mixed_clean, noise, noise_snr_db)
    peak = np.max(np.abs(mixed_noisy))
    if peak > 0.95:
        mixed_noisy = mixed_noisy * 0.95 / peak
    return mixed_noisy, labels


def compute_metrics(result, labels, threshold):
    sims = result["similarities"]
    frame_times = np.arange(result["n_frames"]) * GTCRN_HOP / SAMPLE_RATE
    metrics = {}
    for (s, e, spk) in labels:
        mask = (frame_times >= s) & (frame_times < e)
        seg_sims = sims[mask]
        if len(seg_sims) > 0:
            metrics[f"{spk}_{s:.1f}-{e:.1f}s"] = {
                "mean_similarity": float(np.mean(seg_sims)),
                "std_similarity": float(np.std(seg_sims)),
                "target_ratio": float((seg_sims > threshold).mean()),
                "n_frames": int(len(seg_sims)),
            }
    return metrics


def aggregate_metrics(metrics_by_scenario):
    """從各場景的 metrics 計算匯總指標。"""
    target_ratios = []
    interf_ratios = []
    target_sims = []
    interf_sims = []

    for sc_id, metrics in metrics_by_scenario.items():
        for seg_name, m in metrics.items():
            is_target = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            if is_target:
                target_ratios.append(m["target_ratio"])
                target_sims.append(m["mean_similarity"])
            else:
                interf_ratios.append(m["target_ratio"])
                interf_sims.append(m["mean_similarity"])

    return {
        "target_recall_mean": float(np.mean(target_ratios)) if target_ratios else 0.0,
        "interf_leakage_mean": float(np.mean(interf_ratios)) if interf_ratios else 0.0,
        "target_sim_mean": float(np.mean(target_sims)) if target_sims else 0.0,
        "interf_sim_mean": float(np.mean(interf_sims)) if interf_sims else 0.0,
        "sim_gap": float(np.mean(target_sims) - np.mean(interf_sims)) if target_sims and interf_sims else 0.0,
    }


def aggregate_by_scenario(metrics_by_scenario):
    """各場景分開匯總。"""
    result = {}
    for sc_id, metrics in metrics_by_scenario.items():
        tr_list, ir_list = [], []
        for seg_name, m in metrics.items():
            is_target = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
            if is_target:
                tr_list.append(m["target_ratio"])
            else:
                ir_list.append(m["target_ratio"])
        result[sc_id] = {
            "target_recall": float(np.mean(tr_list)) if tr_list else 0.0,
            "interf_leakage": float(np.mean(ir_list)) if ir_list else 0.0,
        }
    return result


# ══════════════════════════════════════════════════════
# 消融實驗配置矩陣
# ══════════════════════════════════════════════════════

BASELINE = {
    "enrollment_path": ENROLLMENT_CLEAN,
    "threshold": 0.25,
    "pvad_window_sec": 0.5,
    "ema_alpha": 0.6,
    "gain_floor": 0.0,
    "pvad_interval": 32,
    "use_augmented_enrollment": False,
    "denoise_enrollment": False,
}

ABLATION_CONFIGS = [
    ("baseline",           "Baseline (clean enroll, default params)", {}),
    ("noisy_enroll",       "Noisy enrollment (SNR 10dB)",            {"enrollment_path": ENROLLMENT_NOISY}),
    ("augmented_enroll",   "Augmented enrollment (5-noise centroid)", {"use_augmented_enrollment": True}),
    ("threshold_020",      "Threshold = 0.20",                       {"threshold": 0.20}),
    ("threshold_030",      "Threshold = 0.30",                       {"threshold": 0.30}),
    ("window_1s",          "pVAD window = 1.0s",                     {"pvad_window_sec": 1.0}),
    ("ema_030",            "EMA alpha = 0.3 (slower)",               {"ema_alpha": 0.3}),
    ("ema_090",            "EMA alpha = 0.9 (faster)",               {"ema_alpha": 0.9}),
    ("gain_floor_005",     "Gain floor = 0.05 (-26dB)",              {"gain_floor": 0.05}),
    ("denoise_enroll",     "Denoise enrollment (GTCRN)",             {"denoise_enrollment": True}),
]

SCENARIOS = [
    ("scenario_a", "A: Alternating (0911)"),
    ("scenario_b", "B: Overlap (0911)"),
    ("scenario_c", "C: Noisy Alternating"),
    ("scenario_d", "D: Alternating (FEMH)"),
]


def run_ablation():
    print("=" * 70)
    print("pVAD-SE Pipeline v3 — 消融實驗")
    print(f"配置數: {len(ABLATION_CONFIGS)}, 場景數: {len(SCENARIOS)}")
    print(f"總跑數: {len(ABLATION_CONFIGS) * len(SCENARIOS)}")
    print("=" * 70)

    # 載入模型（共用）
    print("\n載入模型...")
    models_dir = PROJECT_DIR / "models"
    campp_path = models_dir / "campplus" / "campplus.onnx"
    if not campp_path.exists():
        campp_path = models_dir / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    speaker_encoder = SpeakerEncoder(str(campp_path))
    denoiser = GTCRNDenoiser()
    print("  模型載入完成")

    # 載入音訊素材
    print("\n載入音訊素材...")
    hsuan_audio = read_audio(str(HSUAN_SOURCE))
    audio_0911 = read_audio(str(INTERFERER_0911))
    audio_femh = read_audio(str(INTERFERER_FEMH))
    print(f"  hsuan: {len(hsuan_audio)/SAMPLE_RATE:.1f}s")
    print(f"  0911:  {len(audio_0911)/SAMPLE_RATE:.1f}s")
    print(f"  FEMH:  {len(audio_femh)/SAMPLE_RATE:.1f}s")

    # 預建場景音訊（所有配置共用相同的混合音訊）
    print("\n預建場景音訊...")
    scenario_data = {
        "scenario_a": build_scenario_a(hsuan_audio, audio_0911),
        "scenario_b": build_scenario_b(hsuan_audio, audio_0911),
        "scenario_c": build_scenario_c(hsuan_audio, audio_0911),
        "scenario_d": build_scenario_a(hsuan_audio, audio_femh),
    }

    # ── 主實驗迴圈 ────────────────────────────────────
    all_results = {}
    total_runs = len(ABLATION_CONFIGS) * len(SCENARIOS)
    run_idx = 0
    t_total = time.time()

    for cfg_id, cfg_name, overrides in ABLATION_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Config: {cfg_name}")
        print(f"{'=' * 60}")

        # 合併 baseline + overrides
        config = {**BASELINE, **overrides}
        metrics_by_scenario = {}

        for sc_id, sc_title in SCENARIOS:
            run_idx += 1
            mixed, labels = scenario_data[sc_id]

            print(f"  [{run_idx}/{total_runs}] {sc_title}...", end=" ", flush=True)
            t0 = time.time()

            result = run_parallel_pipeline_from_audio(
                enrollment_path=config["enrollment_path"],
                input_audio=mixed,
                denoiser=denoiser,
                speaker_encoder=speaker_encoder,
                threshold=config["threshold"],
                gain_floor=config["gain_floor"],
                pvad_interval=config["pvad_interval"],
                pvad_window_sec=config["pvad_window_sec"],
                ema_alpha=config["ema_alpha"],
                use_augmented_enrollment=config["use_augmented_enrollment"],
                denoise_enrollment=config["denoise_enrollment"],
            )

            elapsed = time.time() - t0

            # 計算 metrics（threshold 用此配置的值）
            metrics = compute_metrics(result, labels, config["threshold"])
            metrics_by_scenario[sc_id] = metrics

            # 儲存音檔
            wav_path = OUTPUT_DIR / f"{cfg_id}_{sc_id}_output.wav"
            write_audio(str(wav_path), result["output_audio"])

            # 印出摘要
            tr_vals = []
            ir_vals = []
            for seg_name, m in metrics.items():
                is_target = "hsuan" in seg_name and "interferer" not in seg_name and "+" not in seg_name
                if is_target:
                    tr_vals.append(m["target_ratio"])
                else:
                    ir_vals.append(m["target_ratio"])
            tr_avg = np.mean(tr_vals) if tr_vals else 0
            ir_avg = np.mean(ir_vals) if ir_vals else 0
            print(f"done ({elapsed:.1f}s) TR={tr_avg:.1%} IR={ir_avg:.1%}")

        # 匯總此配置
        agg = aggregate_metrics(metrics_by_scenario)
        by_sc = aggregate_by_scenario(metrics_by_scenario)

        all_results[cfg_id] = {
            "name": cfg_name,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            "overrides": {k: str(v) if isinstance(v, Path) else v for k, v in overrides.items()},
            "aggregate": agg,
            "by_scenario": by_sc,
            "detailed_metrics": {
                sc: {seg: m for seg, m in mets.items()}
                for sc, mets in metrics_by_scenario.items()
            },
        }

    total_elapsed = time.time() - t_total
    print(f"\n\n{'=' * 60}")
    print(f"消融實驗完成！總耗時: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'=' * 60}")

    # ── 儲存 JSON 報告 ────────────────────────────────
    report_path = OUTPUT_DIR / "ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n報告已存: {report_path}")

    # ── 產生 CSV 匯總表 ───────────────────────────────
    csv_path = OUTPUT_DIR / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # Header
        header = ["Config", "Description", "Changed Param",
                  "Overall TR", "Overall IR", "Sim Gap",
                  "A: TR", "A: IR", "B: TR", "B: IR",
                  "C: TR", "C: IR", "D: TR", "D: IR"]
        writer.writerow(header)

        for cfg_id, cfg_name, overrides in ABLATION_CONFIGS:
            r = all_results[cfg_id]
            agg = r["aggregate"]
            by_sc = r["by_scenario"]
            changed = list(overrides.keys())[0] if overrides else "—"
            row = [
                cfg_id, cfg_name, changed,
                f"{agg['target_recall_mean']:.1%}",
                f"{agg['interf_leakage_mean']:.1%}",
                f"{agg['sim_gap']:.3f}",
            ]
            for sc in ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]:
                sc_data = by_sc.get(sc, {})
                row.append(f"{sc_data.get('target_recall', 0):.1%}")
                row.append(f"{sc_data.get('interf_leakage', 0):.1%}")
            writer.writerow(row)

    print(f"CSV 匯總表: {csv_path}")

    # ── 產生熱力圖 ────────────────────────────────────
    generate_heatmap(all_results)

    # ── 印出匯總表 ────────────────────────────────────
    print_summary_table(all_results)

    return all_results


def generate_heatmap(all_results):
    """產生消融實驗熱力圖。"""
    configs = list(all_results.keys())
    n_configs = len(configs)

    # 準備數據矩陣
    # 列: 配置, 欄: [A-TR, A-IR, B-TR, B-IR, C-TR, C-IR, D-TR, D-IR]
    col_labels = ["A:TR", "A:IR", "B:TR", "B:IR", "C:TR", "C:IR", "D:TR", "D:IR"]
    data = np.zeros((n_configs, len(col_labels)))

    row_labels = []
    for i, cfg_id in enumerate(configs):
        r = all_results[cfg_id]
        row_labels.append(r["name"][:35])
        by_sc = r["by_scenario"]
        for j, sc in enumerate(["scenario_a", "scenario_b", "scenario_c", "scenario_d"]):
            sc_data = by_sc.get(sc, {})
            data[i, j*2] = sc_data.get("target_recall", 0) * 100
            data[i, j*2+1] = sc_data.get("interf_leakage", 0) * 100

    fig, ax = plt.subplots(figsize=(14, 8))

    # 自定義顏色映射
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(n_configs))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=9)

    # 文字標註
    for i in range(n_configs):
        for j in range(len(col_labels)):
            val = data[i, j]
            is_ir = "IR" in col_labels[j]
            # IR 欄位：低值好（綠），高值差（紅）
            # TR 欄位：高值好（綠），低值差（紅）
            color = "white" if (val > 70 or val < 30) else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    # IR 欄位用反轉的顏色指示（標記）
    for j, label in enumerate(col_labels):
        if "IR" in label:
            ax.text(j, -0.7, "↓better", ha="center", va="center",
                    color="red", fontsize=7, fontstyle="italic")
        else:
            ax.text(j, -0.7, "↑better", ha="center", va="center",
                    color="green", fontsize=7, fontstyle="italic")

    ax.set_title("Ablation Study: Target Recall (TR) & Interferer Leakage (IR) by Scenario",
                 fontsize=13, fontweight="bold", pad=20)
    plt.colorbar(im, ax=ax, label="Percentage (%)", shrink=0.8)
    plt.tight_layout()

    heatmap_path = OUTPUT_DIR / "ablation_heatmap.png"
    plt.savefig(str(heatmap_path), dpi=150)
    plt.close()
    print(f"熱力圖: {heatmap_path}")


def print_summary_table(all_results):
    """在終端印出漂亮的匯總表。"""
    print(f"\n{'=' * 90}")
    print(f"{'消融實驗匯總':^90}")
    print(f"{'=' * 90}")
    print(f"{'配置':36s} {'整體TR':>8s} {'整體IR':>8s} {'Sim差距':>8s} {'C:TR':>7s} {'C:IR':>7s}")
    print(f"{'-' * 90}")

    for cfg_id in all_results:
        r = all_results[cfg_id]
        agg = r["aggregate"]
        c_data = r["by_scenario"].get("scenario_c", {})
        name = r["name"][:35]
        print(f"{name:36s} "
              f"{agg['target_recall_mean']:7.1%} "
              f"{agg['interf_leakage_mean']:7.1%} "
              f"{agg['sim_gap']:8.3f} "
              f"{c_data.get('target_recall', 0):6.1%} "
              f"{c_data.get('interf_leakage', 0):6.1%}")

    print(f"{'=' * 90}")
    print("TR = Target Recall (越高越好), IR = Interferer Leakage (越低越好)")
    print("Sim差距 = target sim - interferer sim (越大越好)")


if __name__ == "__main__":
    run_ablation()
