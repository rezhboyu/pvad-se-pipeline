#!/usr/bin/env python3
"""
Personal VAD 訓練腳本
=====================
基於 Google Personal VAD 架構，訓練小型 LSTM 模型：
  輸入: Fbank(80) + enrollment d-vector(192) = 272 維/幀
  輸出: 3 類（non_speech, target, non_target）逐幀機率

訓練資料: TSE-AISHELL（400 說話者, 4508 train + 492 val）

用法:
    python train_personal_vad.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder, _compute_fbank

# ── 常數 ──────────────────────────────────────────────
FBANK_HOP = 160          # 10ms hop
FBANK_WIN = 400          # 25ms window
ENERGY_FLOOR = 0.01      # RMS threshold for speech
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = Path(os.path.expanduser("~/Desktop/datasets_copy/tse_aishell"))
MODEL_OUTPUT_DIR = PROJECT_DIR / "models" / "personal_vad"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 標籤生成 ──────────────────────────────────────────
def generate_frame_labels(mixture: np.ndarray, target: np.ndarray,
                          hop: int = FBANK_HOP, win: int = FBANK_WIN,
                          energy_floor: float = ENERGY_FLOOR) -> np.ndarray:
    """
    從 mixture 和 target 音訊生成逐幀標籤。

    Returns: (n_frames,) int64 array
        0 = non_speech (靜音)
        1 = target speaker
        2 = non_target speaker (interferer)
    """
    n_frames = (len(mixture) - win) // hop + 1
    labels = np.zeros(n_frames, dtype=np.int64)

    for i in range(n_frames):
        start = i * hop
        end = start + win
        mix_frame = mixture[start:end]
        tgt_frame = target[start:end]

        mix_rms = np.sqrt(np.mean(mix_frame ** 2) + 1e-12)
        tgt_rms = np.sqrt(np.mean(tgt_frame ** 2) + 1e-12)

        if mix_rms < energy_floor:
            labels[i] = 0  # non_speech
        elif tgt_rms >= energy_floor * 0.5:
            # target 有能量 → target speech
            labels[i] = 1
        else:
            # mixture 有能量但 target 沒有 → non-target speech
            labels[i] = 2

    return labels


# ── Dataset ───────────────────────────────────────────
class PersonalVADDataset(Dataset):
    def __init__(self, samples_json: str, speaker_encoder: SpeakerEncoder,
                 max_samples: int = None):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.encoder = speaker_encoder

        # 預處理：提取所有特徵和標籤
        print(f"  預處理 {len(self.samples)} 個樣本...")
        self.data = []
        t0 = time.time()

        for idx, sample in enumerate(self.samples):
            sample_dir = self._resolve_path(sample["path"])
            if not sample_dir.exists():
                continue

            try:
                enrollment_path = sample_dir / "enrollment.wav"
                mixture_path = sample_dir / "mixture.wav"
                target_path = sample_dir / "target.wav"

                # 讀取音訊
                import soundfile as sf
                enrollment, _ = sf.read(str(enrollment_path), dtype="float32")
                mixture, _ = sf.read(str(mixture_path), dtype="float32")
                target, _ = sf.read(str(target_path), dtype="float32")

                # 提取 enrollment d-vector (192 dim with CAM++)
                dvector = self.encoder.extract_embedding(enrollment)

                # 計算 mixture 的 Fbank 特徵
                fbank = _compute_fbank(mixture)  # (T, 80)

                # 生成標籤
                labels = generate_frame_labels(mixture, target)

                # 確保 fbank 和 labels 長度一致
                min_len = min(len(fbank), len(labels))
                fbank = fbank[:min_len]
                labels = labels[:min_len]

                # 每幀拼接 d-vector
                dvector_repeated = np.tile(dvector, (min_len, 1))  # (T, 192)
                features = np.concatenate([fbank, dvector_repeated], axis=1)  # (T, 272)

                self.data.append({
                    "features": features.astype(np.float32),
                    "labels": labels,
                    "speaker_id": sample.get("speaker_id", "unknown"),
                })

            except Exception as e:
                if idx < 5:
                    print(f"    [WARN] sample {idx}: {e}")
                continue

            if (idx + 1) % 500 == 0:
                print(f"    已處理 {idx+1}/{len(self.samples)} "
                      f"({time.time()-t0:.0f}s)")

        print(f"  完成: {len(self.data)} 個有效樣本 ({time.time()-t0:.0f}s)")

    def _resolve_path(self, path_str: str) -> Path:
        """解析路徑，支援 datasets_copy 和原始路徑。"""
        p = Path(path_str)
        if p.exists():
            return p
        # 嘗試 datasets_copy 路徑
        alt = Path(path_str.replace(
            "VOICE\\datasets\\tse_aishell",
            "datasets_copy\\tse_aishell"
        ).replace(
            "VOICE/datasets/tse_aishell",
            "datasets_copy/tse_aishell"
        ))
        if alt.exists():
            return alt
        # 嘗試用 DATASET_DIR
        sample_name = p.name
        parent_name = p.parent.name  # train or val
        alt2 = DATASET_DIR / parent_name / sample_name
        if alt2.exists():
            return alt2
        return p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.from_numpy(d["features"]),
            torch.from_numpy(d["labels"]),
        )


def collate_fn(batch):
    """動態 padding 到 batch 內最長序列。"""
    features, labels = zip(*batch)
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)

    padded_features = torch.zeros(len(features), max_len, features[0].shape[1])
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)

    for i, (f, l) in enumerate(zip(features, labels)):
        padded_features[i, :f.shape[0]] = f
        padded_labels[i, :l.shape[0]] = l

    return padded_features, padded_labels, torch.tensor(lengths)


# ── 模型 ──────────────────────────────────────────────
class PersonalVADModel(nn.Module):
    """
    Personal VAD: 2-layer LSTM + FC head.
    輸入: (batch, seq_len, 272)  [Fbank(80) + d-vector(192)]
    輸出: (batch, seq_len, 3)    [non_speech, target, non_target]
    """
    def __init__(self, input_dim=272, hidden_dim=64, n_classes=3, n_layers=2,
                 dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, 272)
        out, _ = self.lstm(x)  # (B, T, 64)
        out = self.fc1(out)    # (B, T, 64)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)    # (B, T, 3)
        return out


# ── 訓練 ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_frames = 0

    for features, labels, lengths in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)  # (B, T, 3)

        # Mask padding
        mask = torch.arange(logits.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)

        # Flatten for loss
        logits_flat = logits[mask]         # (N, 3)
        labels_flat = labels[mask]         # (N,)

        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * labels_flat.numel()
        preds = logits_flat.argmax(dim=1)
        total_correct += (preds == labels_flat).sum().item()
        total_frames += labels_flat.numel()

    return total_loss / total_frames, total_correct / total_frames


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_frames = 0

    # Per-class accuracy
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for features, labels, lengths in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            mask = torch.arange(logits.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)

            logits_flat = logits[mask]
            labels_flat = labels[mask]

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item() * labels_flat.numel()

            preds = logits_flat.argmax(dim=1)
            total_correct += (preds == labels_flat).sum().item()
            total_frames += labels_flat.numel()

            for c in range(3):
                c_mask = labels_flat == c
                class_total[c] += c_mask.sum().item()
                class_correct[c] += ((preds == labels_flat) & c_mask).sum().item()

    class_acc = [
        class_correct[c] / max(class_total[c], 1) for c in range(3)
    ]
    return (
        total_loss / total_frames,
        total_correct / total_frames,
        class_acc,
        class_total,
    )


def export_onnx(model, output_path, input_dim=272):
    """匯出 ONNX 模型（支援動態序列長度）。"""
    model.eval()
    model.cpu()

    dummy = torch.randn(1, 100, input_dim)

    # 使用 dynamo=False 避免新版 PyTorch 的 dynamic shape 問題
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
        dynamo=False,
    )
    file_size = os.path.getsize(output_path) / 1024
    print(f"  ONNX export: {output_path} ({file_size:.0f} KB)")


# ── 主程式 ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Personal VAD 訓練")
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATASET_DIR}")
    print("=" * 60)

    # 載入 speaker encoder
    print("\n載入 CAM++ speaker encoder...")
    campp_path = PROJECT_DIR / "models" / "campplus" / "campplus.onnx"
    speaker_encoder = SpeakerEncoder(str(campp_path))
    print(f"  embed_dim: {speaker_encoder.embed_dim}")

    input_dim = 80 + speaker_encoder.embed_dim  # Fbank + d-vector
    print(f"  input_dim: {input_dim} (Fbank 80 + d-vector {speaker_encoder.embed_dim})")

    # 載入資料集
    print("\n載入訓練資料...")
    train_dataset = PersonalVADDataset(
        str(DATASET_DIR / "train_samples.json"),
        speaker_encoder,
    )
    print("\n載入驗證資料...")
    val_dataset = PersonalVADDataset(
        str(DATASET_DIR / "val_samples.json"),
        speaker_encoder,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 統計標籤分佈
    all_labels = np.concatenate([d["labels"] for d in train_dataset.data])
    label_names = ["non_speech", "target", "non_target"]
    print("\n標籤分佈:")
    for c in range(3):
        count = (all_labels == c).sum()
        pct = count / len(all_labels) * 100
        print(f"  {label_names[c]:12s}: {count:>8d} ({pct:.1f}%)")

    # 計算 class weights（處理不平衡）
    class_counts = np.array([(all_labels == c).sum() for c in range(3)], dtype=np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3  # normalize
    print(f"  class weights: {class_weights}")
    weights_tensor = torch.from_numpy(class_weights).to(DEVICE)

    # 建立模型
    model = PersonalVADModel(input_dim=input_dim, hidden_dim=64, n_classes=3)
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型參數量: {n_params:,} ({n_params/1000:.1f}K)")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # 訓練迴圈
    n_epochs = 30
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    max_patience = 8

    print(f"\n開始訓練 ({n_epochs} epochs)...")
    print(f"{'Epoch':>5s} {'Train Loss':>11s} {'Train Acc':>10s} {'Val Loss':>10s} {'Val Acc':>9s} {'NS Acc':>7s} {'TG Acc':>7s} {'NT Acc':>7s} {'Time':>6s}")
    print("-" * 85)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, class_acc, class_total = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # 儲存最佳模型
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "best_model.pt")
            marker = " *"
        else:
            patience_counter += 1

        print(f"{epoch:5d} {train_loss:11.4f} {train_acc:10.1%} {val_loss:10.4f} {val_acc:9.1%} "
              f"{class_acc[0]:7.1%} {class_acc[1]:7.1%} {class_acc[2]:7.1%} {elapsed:5.1f}s{marker}")

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    # 載入最佳模型
    print(f"\n載入最佳模型 (epoch {best_epoch})...")
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pt", weights_only=True))

    # 最終驗證
    val_loss, val_acc, class_acc, class_total = validate(model, val_loader, criterion, DEVICE)
    print(f"\n最終驗證結果:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.1%}")
    for c in range(3):
        print(f"  {label_names[c]:12s}: {class_acc[c]:.1%} ({class_total[c]} frames)")

    # 匯出 ONNX
    print("\n匯出 ONNX 模型...")
    onnx_path = MODEL_OUTPUT_DIR / "personal_vad.onnx"
    export_onnx(model, onnx_path, input_dim=input_dim)

    # 驗證 ONNX
    print("\n驗證 ONNX 模型...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 50, input_dim).astype(np.float32)
    onnx_output = sess.run(None, {"features": dummy_input})[0]
    print(f"  輸入: {dummy_input.shape}")
    print(f"  輸出: {onnx_output.shape}")
    print(f"  輸出範例: {onnx_output[0, 0, :]}")

    # PyTorch vs ONNX 一致性
    model.cpu()
    model.eval()
    with torch.no_grad():
        pt_output = model(torch.from_numpy(dummy_input)).numpy()
    diff = np.abs(pt_output - onnx_output).max()
    print(f"  PyTorch vs ONNX max diff: {diff:.6f}")

    print(f"\n訓練完成！模型已存: {onnx_path}")


if __name__ == "__main__":
    main()
