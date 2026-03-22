#!/usr/bin/env python3
"""
下載預訓練模型
==============
1. DTLN：從 GitHub 下載預訓練 Keras (.h5) 模型
2. ECAPA-TDNN：從 HuggingFace / SpeechBrain 下載預訓練模型

下載後存放於 models/ 資料夾。
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── DTLN 下載 ─────────────────────────────────────────
def download_dtln():
    """
    從 DTLN GitHub 下載預訓練模型。
    DTLN 提供 .h5 格式的 Keras 模型，之後用 export_onnx.py 轉為 ONNX。
    也可以直接下載已匯出的 ONNX 模型（若 release 有提供）。
    """
    dtln_dir = MODELS_DIR / "dtln"
    dtln_dir.mkdir(exist_ok=True)

    # DTLN 預訓練模型 URL（從 GitHub repo 取得）
    # 方式 A：下載 .h5 格式（需要 export_onnx.py 轉換）
    h5_url = (
        "https://github.com/breizhn/DTLN/raw/master/pretrained_model/"
        "model.h5"
    )
    h5_path = dtln_dir / "dtln_model.h5"

    # 方式 B：直接下載已匯出的 ONNX 模型（如果 release 提供的話）
    # DTLN 作者在 saved_model/ 中提供了 ONNX 版本
    onnx_urls = {
        "dtln_1.onnx": (
            "https://github.com/breizhn/DTLN/raw/master/pretrained_model/"
            "model_1.onnx"
        ),
        "dtln_2.onnx": (
            "https://github.com/breizhn/DTLN/raw/master/pretrained_model/"
            "model_2.onnx"
        ),
    }

    print("=" * 60)
    print("下載 DTLN 預訓練模型...")
    print("=" * 60)

    # 先嘗試下載 ONNX 版本
    onnx_ok = True
    for filename, url in onnx_urls.items():
        dest = dtln_dir / filename
        if dest.exists():
            print(f"  [跳過] {filename} 已存在")
            continue
        try:
            print(f"  下載 {filename} ...")
            urllib.request.urlretrieve(url, str(dest))
            print(f"  [完成] {dest}")
        except Exception as e:
            print(f"  [警告] 無法下載 ONNX 版本 ({e})，將嘗試下載 .h5 版本")
            onnx_ok = False
            break

    if not onnx_ok:
        # 退而求其次，下載 .h5 並之後轉換
        if h5_path.exists():
            print(f"  [跳過] {h5_path.name} 已存在")
        else:
            try:
                print(f"  下載 dtln_model.h5 ...")
                urllib.request.urlretrieve(h5_url, str(h5_path))
                print(f"  [完成] {h5_path}")
                print("  ⚠ 需要執行 export_onnx.py 將 .h5 轉為 ONNX")
            except Exception as e:
                print(f"  [錯誤] 下載失敗：{e}")
                print("  請手動從 https://github.com/breizhn/DTLN 下載模型")
                return False

    print()
    return True


# ── ECAPA-TDNN 下載 ───────────────────────────────────
def download_ecapa_tdnn():
    """
    使用 SpeechBrain 下載 ECAPA-TDNN 預訓練模型。
    模型來自 HuggingFace: speechbrain/spkrec-ecapa-voxceleb

    注意：此步驟需要 PyTorch 和 SpeechBrain（僅下載時需要）。
    """
    ecapa_dir = MODELS_DIR / "ecapa_tdnn"
    ecapa_dir.mkdir(exist_ok=True)

    # 檢查是否已有 ONNX 版本
    onnx_path = ecapa_dir / "ecapa_tdnn.onnx"
    if onnx_path.exists():
        print("=" * 60)
        print("ECAPA-TDNN ONNX 模型已存在，跳過下載")
        print("=" * 60)
        return True

    print("=" * 60)
    print("下載 ECAPA-TDNN 預訓練模型 (SpeechBrain)...")
    print("=" * 60)

    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("  [錯誤] 需要 PyTorch 和 SpeechBrain")
        print("  pip install torch torchaudio speechbrain")
        return False

    try:
        # SpeechBrain 會自動從 HuggingFace 下載並快取模型
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ecapa_dir / "speechbrain_cache"),
        )
        print(f"  [完成] 模型已下載至 {ecapa_dir / 'speechbrain_cache'}")
        print("  ⚠ 需要執行 export_onnx.py 將模型轉為 ONNX")

        # 保存參考資訊
        info_path = ecapa_dir / "model_info.txt"
        with open(info_path, "w") as f:
            f.write("Model: ECAPA-TDNN\n")
            f.write("Source: speechbrain/spkrec-ecapa-voxceleb\n")
            f.write("Embedding dim: 192\n")
            f.write("Training data: VoxCeleb1+2\n")

    except Exception as e:
        print(f"  [錯誤] 下載失敗：{e}")
        return False

    print()
    return True


# ── 主程式 ────────────────────────────────────────────
def main():
    print("\n🔽 開始下載預訓練模型\n")

    ok_dtln = download_dtln()
    ok_ecapa = download_ecapa_tdnn()

    print("=" * 60)
    if ok_dtln and ok_ecapa:
        print("✅ 所有模型下載完成！")
        print("   下一步：執行 python export_onnx.py 匯出 ONNX 模型")
    else:
        print("⚠ 部分模型下載失敗，請檢查上方錯誤訊息")
    print("=" * 60)


if __name__ == "__main__":
    main()
