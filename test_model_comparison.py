#!/usr/bin/env python3
"""
三模型 × 多窗口 × EMA 對比測試
================================
比較 CAM++、ECAPA-TDNN、WeSpeaker 在不同窗口和 EMA 設定下的表現。

測試維度：
1. 模型：CAM++ / ECAPA-TDNN / WeSpeaker ResNet34
2. 窗口：0.5s / 0.75s / 1.0s
3. EMA alpha：0.2 (強平滑) / 0.3 (中) / 0.5 (弱平滑) / 1.0 (無EMA)
4. 場景：Clean / Noisy 10dB / Noisy 5dB
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

# ── 路徑 ──
PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "models"
MAT = Path.home() / "Desktop" / "VOICE" / "MAT"
OUT = PROJECT / "test_model_compare"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT))
from utils.audio import SAMPLE_RATE, read_audio
from utils.speaker_encoder import SpeakerEncoder, cosine_similarity, _compute_fbank

import onnxruntime as ort


# ═══════════════════════════════════════════════════════════
# ECAPA-TDNN Wrapper (不同輸入格式: batch, 80, T)
# ═══════════════════════════════════════════════════════════
class EcapaTDNNEncoder:
    """SpeechBrain ECAPA-TDNN ONNX wrapper.

    輸入格式: (batch, 80, T) — 注意和 WeSpeaker/CAM++ 相反
    輸出: (batch, 192) L2-normalized embedding
    """

    def __init__(self, onnx_path: str):
        self.onnx_path = Path(onnx_path)
        assert self.onnx_path.exists(), f"ONNX 不存在: {onnx_path}"

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 確認輸入格式
        inp_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        self.embed_dim = out_shape[-1] if isinstance(out_shape[-1], int) else 192
        print(f"  ECAPA-TDNN: input={inp_shape}, output={out_shape}, embed_dim={self.embed_dim}")

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        fbank = _compute_fbank(audio)  # (T, 80)
        # ECAPA-TDNN 要 (batch, 80, T) — 轉置
        fbank_batch = fbank.T[np.newaxis, :, :]  # (1, 80, T)

        embedding = self.session.run(
            [self.output_name],
            {self.input_name: fbank_batch}
        )[0].squeeze()

        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# EMA 模擬器
# ═══════════════════════════════════════════════════════════
def compute_similarities_with_ema(encoder, enroll_emb, audio, window_sec, ema_alpha):
    """模擬 pVAD 的逐窗口提取 + EMA 平滑。"""
    win_samples = int(window_sec * SAMPLE_RATE)
    n_windows = len(audio) // win_samples

    if n_windows == 0:
        return [], []

    raw_sims = []
    ema_sims = []
    ema_val = None

    for i in range(min(n_windows, 30)):  # 最多 30 窗口
        chunk = audio[i * win_samples : (i + 1) * win_samples]
        emb = encoder.extract_embedding(chunk)
        raw_sim = cosine_similarity(emb, enroll_emb)
        raw_sims.append(raw_sim)

        if ema_val is None:
            ema_val = raw_sim
        else:
            ema_val = ema_alpha * raw_sim + (1 - ema_alpha) * ema_val
        ema_sims.append(ema_val)

    return raw_sims, ema_sims


def add_noise(audio, snr_db):
    rms_s = np.sqrt(np.mean(audio**2) + 1e-12)
    noise = np.random.randn(len(audio)).astype(np.float32)
    rms_n = np.sqrt(np.mean(noise**2) + 1e-12)
    scale = rms_s / (rms_n * 10**(snr_db/20))
    return (audio + noise * scale).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("三模型 × 多窗口 × EMA 對比測試")
    print("=" * 80)

    # ── 載入音檔 ──
    print("\n[1] 載入音檔...")
    hsuan_enroll = read_audio(str(MAT / "275eaceb-2387-4f9e-aef5-1e9996b8f024_hsuan_7.wav"))
    hsuan_source = read_audio(str(MAT / "67e84de3-84e4-4327-b2e3-51f0f8341131_hsuan_7.wav"))
    audio_0911   = read_audio(str(MAT / "5e4564bd-c932-4391-bbc9-e01981d98ab9_0911636193_7.wav"))
    print(f"  hsuan enroll: {len(hsuan_enroll)/SAMPLE_RATE:.1f}s")
    print(f"  hsuan source: {len(hsuan_source)/SAMPLE_RATE:.1f}s")
    print(f"  0911 other:   {len(audio_0911)/SAMPLE_RATE:.1f}s")

    # 加噪版本
    np.random.seed(42)
    scenarios = {
        "clean":    (hsuan_source, audio_0911),
        "noisy_10": (add_noise(hsuan_source, 10), add_noise(audio_0911, 10)),
        "noisy_5":  (add_noise(hsuan_source, 5),  add_noise(audio_0911, 5)),
    }

    # ── 載入模型 ──
    print("\n[2] 載入模型...")
    models = {}

    campp_path = MODELS / "campplus" / "campplus.onnx"
    if campp_path.exists():
        models["CAM++"] = SpeakerEncoder(str(campp_path))
        print(f"  CAM++ loaded (embed_dim={models['CAM++'].embed_dim})")

    ecapa_path = MODELS / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    if ecapa_path.exists():
        try:
            models["ECAPA-TDNN"] = EcapaTDNNEncoder(str(ecapa_path))
        except Exception as e:
            print(f"  [WARN] ECAPA-TDNN 載入失敗: {e}")

    wespeaker_path = MODELS / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if wespeaker_path.exists():
        models["WeSpeaker"] = SpeakerEncoder(str(wespeaker_path))
        print(f"  WeSpeaker loaded (embed_dim={models['WeSpeaker'].embed_dim})")

    if not models:
        print("沒有可用的模型！")
        return

    print(f"\n  可用模型: {list(models.keys())}")

    # ── 提取 enrollment embedding ──
    print("\n[3] 提取 enrollment embeddings...")
    enroll_embs = {}
    for name, enc in models.items():
        enroll_embs[name] = enc.extract_embedding(hsuan_enroll)
        print(f"  {name}: dim={enroll_embs[name].shape}")

    # ── 測試矩陣 ──
    windows = [0.5, 0.75, 1.0]
    ema_alphas = [0.2, 0.3, 0.5, 1.0]  # 1.0 = no EMA

    results = {}

    print("\n[4] 開始對比測試...")
    print(f"    模型: {list(models.keys())}")
    print(f"    窗口: {windows}")
    print(f"    EMA alpha: {ema_alphas}")
    print(f"    場景: {list(scenarios.keys())}")
    total = len(models) * len(windows) * len(ema_alphas) * len(scenarios)
    print(f"    總組合數: {total}")

    count = 0
    for model_name, encoder in models.items():
        enroll_emb = enroll_embs[model_name]

        for win_sec in windows:
            for ema_alpha in ema_alphas:
                for scene_name, (h_audio, o_audio) in scenarios.items():
                    count += 1

                    # hsuan similarity
                    h_raw, h_ema = compute_similarities_with_ema(
                        encoder, enroll_emb, h_audio, win_sec, ema_alpha
                    )
                    # 0911 similarity
                    o_raw, o_ema = compute_similarities_with_ema(
                        encoder, enroll_emb, o_audio, win_sec, ema_alpha
                    )

                    if h_ema and o_ema:
                        h_mean = np.mean(h_ema)
                        o_mean = np.mean(o_ema)
                        gap = h_mean - o_mean
                        h_std = np.std(h_ema)
                        o_std = np.std(o_ema)

                        # EER-approximate threshold
                        eer_th = (h_mean * o_std + o_mean * h_std) / (h_std + o_std + 1e-12)

                        # 計算在不同 threshold 下的錯誤率
                        h_arr = np.array(h_ema)
                        o_arr = np.array(o_ema)

                        # False rejection (hsuan 被拒) & False acceptance (0911 被接受)
                        best_acc = 0
                        best_th = 0
                        for th in np.arange(0.05, 0.60, 0.01):
                            fr = np.mean(h_arr < th)  # hsuan 被拒
                            fa = np.mean(o_arr >= th)  # 0911 被接受
                            acc = 1 - (fr + fa) / 2
                            if acc > best_acc:
                                best_acc = acc
                                best_th = th

                        key = f"{model_name}|{win_sec}|{ema_alpha}|{scene_name}"
                        results[key] = {
                            "model": model_name,
                            "window": win_sec,
                            "ema_alpha": ema_alpha,
                            "scene": scene_name,
                            "hsuan_mean": round(float(h_mean), 4),
                            "hsuan_std": round(float(h_std), 4),
                            "o911_mean": round(float(o_mean), 4),
                            "o911_std": round(float(o_std), 4),
                            "gap": round(float(gap), 4),
                            "eer_threshold": round(float(eer_th), 4),
                            "best_accuracy": round(float(best_acc), 4),
                            "best_threshold": round(float(best_th), 2),
                        }

                    if count % 10 == 0:
                        print(f"  進度: {count}/{total}")

    # ── 結果排名 ──
    print("\n" + "=" * 120)
    print("結果排名（按 gap 排序，分場景）")
    print("=" * 120)

    for scene in ["clean", "noisy_10", "noisy_5"]:
        scene_results = [(k, v) for k, v in results.items() if v["scene"] == scene]
        scene_results.sort(key=lambda x: x[1]["gap"], reverse=True)

        print(f"\n{'─'*120}")
        print(f"場景: {scene}")
        print(f"{'─'*120}")
        print(f"{'Rank':>4} | {'Model':>12} | {'Win':>5} | {'EMA':>5} | "
              f"{'hsuan':>7} | {'0911':>7} | {'GAP':>7} | {'BestAcc':>7} | {'BestTh':>6} | {'Delay':>6}")
        print(f"{'─'*120}")

        for rank, (key, r) in enumerate(scene_results[:10], 1):
            delay = int(r["window"] * 1000)
            ema_str = f"{r['ema_alpha']:.1f}" if r["ema_alpha"] < 1.0 else "none"
            print(f"{rank:>4} | {r['model']:>12} | {r['window']:>4.2f}s | {ema_str:>5} | "
                  f"{r['hsuan_mean']:>7.3f} | {r['o911_mean']:>7.3f} | {r['gap']:>7.3f} | "
                  f"{r['best_accuracy']:>6.1%} | {r['best_threshold']:>5.2f} | {delay:>5}ms")

    # ── 最佳配置推薦 ──
    print("\n" + "=" * 120)
    print("最佳配置推薦（每個延遲等級的最佳組合）")
    print("=" * 120)

    for target_delay in [500, 750, 1000]:
        win_sec = target_delay / 1000
        print(f"\n  延遲 ≤ {target_delay}ms (窗口 {win_sec}s):")

        for scene in ["clean", "noisy_10", "noisy_5"]:
            candidates = [(k, v) for k, v in results.items()
                         if v["scene"] == scene and v["window"] == win_sec]
            if candidates:
                best = max(candidates, key=lambda x: x[1]["gap"])
                r = best[1]
                ema_str = f"alpha={r['ema_alpha']:.1f}" if r["ema_alpha"] < 1.0 else "no EMA"
                print(f"    {scene:>10}: {r['model']:>12} ({ema_str}) "
                      f"gap={r['gap']:.3f} acc={r['best_accuracy']:.1%} th={r['best_threshold']:.2f}")

    # ── EMA 效果分析 ──
    print("\n" + "=" * 120)
    print("EMA 效果分析（同模型+窗口，比較有無 EMA）")
    print("=" * 120)

    for model_name in models:
        for win_sec in windows:
            no_ema_key = f"{model_name}|{win_sec}|1.0|noisy_10"
            if no_ema_key not in results:
                continue
            no_ema = results[no_ema_key]

            best_ema = None
            best_ema_gap = 0
            for alpha in [0.2, 0.3, 0.5]:
                ema_key = f"{model_name}|{win_sec}|{alpha}|noisy_10"
                if ema_key in results and results[ema_key]["gap"] > best_ema_gap:
                    best_ema = results[ema_key]
                    best_ema_gap = results[ema_key]["gap"]

            if best_ema:
                improvement = best_ema["gap"] - no_ema["gap"]
                print(f"  {model_name:>12} {win_sec:.2f}s: "
                      f"no_EMA gap={no_ema['gap']:.3f} → "
                      f"EMA(α={best_ema['ema_alpha']:.1f}) gap={best_ema['gap']:.3f} "
                      f"({'↑' if improvement > 0 else '↓'}{abs(improvement):.3f})")

    # ── 存 JSON ──
    with open(str(OUT / "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已存到: {OUT / 'comparison_results.json'}")

    # ── 畫圖 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        colors = {"CAM++": "#2196F3", "ECAPA-TDNN": "#FF5722", "WeSpeaker": "#4CAF50"}
        markers = {0.2: "o", 0.3: "s", 0.5: "D", 1.0: "x"}

        for ax, scene in zip(axes, ["clean", "noisy_10", "noisy_5"]):
            ax.set_title(f"Scene: {scene}", fontsize=12)
            ax.set_xlabel("Window (s)")
            ax.set_ylabel("Gap (hsuan - 0911)")
            ax.grid(True, alpha=0.3)

            for model_name in models:
                for alpha in ema_alphas:
                    gaps = []
                    ws = []
                    for win_sec in windows:
                        key = f"{model_name}|{win_sec}|{alpha}|{scene}"
                        if key in results:
                            gaps.append(results[key]["gap"])
                            ws.append(win_sec)

                    if gaps:
                        label = f"{model_name} α={alpha:.1f}" if alpha < 1.0 else f"{model_name} noEMA"
                        ax.plot(ws, gaps,
                               color=colors.get(model_name, "gray"),
                               marker=markers.get(alpha, "o"),
                               label=label,
                               alpha=0.8 if alpha in [0.3, 1.0] else 0.4,
                               linewidth=2 if alpha == 0.3 else 1)

            ax.legend(fontsize=7, loc="lower right")

        plt.tight_layout()
        plt.savefig(str(OUT / "model_comparison.png"), dpi=150)
        print(f"圖表已存到: {OUT / 'model_comparison.png'}")

        # ── 第二張圖：EMA 平滑效果視覺化 ──
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8))

        # 找出最佳模型在 noisy_10 場景下的 raw vs EMA similarity 曲線
        for col, win_sec in enumerate(windows):
            for row, (speaker_label, audio_data) in enumerate([
                ("hsuan", scenarios["noisy_10"][0]),
                ("0911",  scenarios["noisy_10"][1]),
            ]):
                ax = axes2[row, col]
                ax.set_title(f"{speaker_label} | win={win_sec}s | noisy_10dB", fontsize=10)
                ax.set_xlabel("Window #")
                ax.set_ylabel("Similarity")
                ax.grid(True, alpha=0.3)

                for model_name, encoder in models.items():
                    enroll_emb = enroll_embs[model_name]

                    # Raw (no EMA)
                    raw_sims, _ = compute_similarities_with_ema(
                        encoder, enroll_emb, audio_data, win_sec, 1.0
                    )
                    # EMA 0.3
                    _, ema_sims = compute_similarities_with_ema(
                        encoder, enroll_emb, audio_data, win_sec, 0.3
                    )

                    if raw_sims:
                        ax.plot(raw_sims, color=colors.get(model_name, "gray"),
                               alpha=0.3, linewidth=1, linestyle="--")
                        ax.plot(ema_sims, color=colors.get(model_name, "gray"),
                               alpha=0.9, linewidth=2, label=f"{model_name}")

                ax.legend(fontsize=8)
                # 畫 threshold 線
                ax.axhline(y=0.20, color="red", linestyle=":", alpha=0.5, label="th=0.20")

        fig2.suptitle("Raw (dashed) vs EMA α=0.3 (solid) — Noisy 10dB", fontsize=13)
        plt.tight_layout()
        plt.savefig(str(OUT / "ema_smoothing_effect.png"), dpi=150)
        print(f"EMA 效果圖已存到: {OUT / 'ema_smoothing_effect.png'}")

    except Exception as e:
        print(f"[WARN] 畫圖失敗: {e}")

    print("\n✓ 完成！")


if __name__ == "__main__":
    main()
