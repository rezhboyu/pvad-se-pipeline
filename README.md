# pVAD + SE Pipeline (Phase 1 Prototype)

B 路線 Phase 1：**Personalized VAD + 輕量 Speech Enhancement** 的 Python 原型。

## 架構概覽

```
輸入混合音頻 → [DTLN 降噪] → [pVAD 判定目標說話者] → [Soft Gating] → 輸出
                                      ↑
                            enrollment d-vector
                           (ECAPA-TDNN 提取)
```

- **SE 模組**：DTLN（兩階段 LSTM，幅度域 + 時域）
- **pVAD 模組**：ECAPA-TDNN embedding + cosine similarity（placeholder，之後替換為訓練過的分類器）
- **Gating**：Soft gating with gain floor（attack 5ms / release 50ms）
- **推論引擎**：純 ONNX Runtime（PyTorch / TensorFlow 僅用於模型匯出）

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 下載預訓練模型
python download_models.py

# 3. 匯出 ONNX（如果 download 沒拿到 ONNX 版本）
python export_onnx.py

# 4. 離線處理
python pipeline.py -e enrollment.wav -i mixed.wav -o output.wav

# 5. 串流模擬
python pipeline_streaming.py -e enrollment.wav -i mixed.wav -o output_stream.wav
```

## 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--threshold` | 0.25 | pVAD cosine similarity 閾值 |
| `--gain-floor` | 0.05 | 非目標部分最小增益 (-26 dB) |
| `--attack-ms` | 5 | 增益上升時間 |
| `--release-ms` | 50 | 增益下降時間 |

## 專案結構

```
pvad-se-pipeline/
├── requirements.txt          # 依賴清單（推論 vs 匯出分開標註）
├── README.md
├── download_models.py        # 下載 DTLN + ECAPA-TDNN 預訓練模型
├── export_onnx.py            # 匯出 ONNX（TF→ONNX, PyTorch→ONNX）
├── pipeline.py               # 離線版主管線
├── pipeline_streaming.py     # 串流版管線（模擬即時，含延遲測量）
├── models/
│   ├── dtln/                 # DTLN ONNX 模型
│   └── ecapa_tdnn/           # ECAPA-TDNN ONNX 模型
├── utils/
│   ├── audio.py              # 音頻 I/O、STFT、Mel 濾波器
│   ├── gating.py             # Soft gating with attack/release
│   └── speaker_encoder.py    # ECAPA-TDNN wrapper + SimplePVAD
└── test_audio/               # 測試用音頻
```

## 技術細節

- 取樣率：16 kHz
- 幀長：512 samples (32 ms)
- 幀移：128 samples (8 ms)
- FFT：512 點
- Mel bins：80
- DTLN hidden size：128（2 層 LSTM × 2 stages）
- ECAPA-TDNN embedding：192 維

## 下一步（Phase 2）

- [ ] 訓練真正的 pVAD 分類器替換 cosine similarity placeholder
- [ ] 實作 ONNX stateful 模型（帶 hidden state I/O）
- [ ] 量化（INT8）降低推論延遲
- [ ] 整合至 USEF-TSE 主系統
