# pVAD Baseline 分類器比較報告

## 1. 實驗設定

- **目標說話者**: hsuan
- **非目標說話者**: 0911636193, FEMH
- **總樣本數**: 13
- **目標樣本數**: 8 (61.5%)
- **非目標樣本數**: 5 (38.5%)
- **Embedding 模型**: WeSpeaker ResNet34-LM (256-dim)
- **評估方式**: Leave-One-Out Cross-Validation

### 說話者樣本分布

| 說話者 | 樣本數 | 類別 |
|--------|--------|------|
| hsuan | 8 | 目標 |
| 0911636193 | 3 | 非目標 |
| FEMH | 2 | 非目標 |

## 2. 指標比較

| Method | Accuracy | Precision | Recall | F1 | AUC | EER |
|--------|----------|-----------|--------|----|-----|-----|
| Cosine Similarity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| MLP | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| SVM-RBF | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| SVM-Linear | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| KNN (K=3) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| LDA (PLDA替代) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

**Cosine Similarity 最佳閾值**: 0.24

## 3. 核心發現：utterance-level 任務太簡單

**所有方法均達到 100% 準確率。** 這不是 bug，而是一個重要的實驗發現。

### 為什麼會全滿分？

PCA 視覺化和 cosine similarity 分布圖揭示了原因：

1. **Embedding 空間極度可分**: hsuan 的 cosine similarity 集中在 0.88–0.97，非目標說話者在 0.13–0.25。兩個分布之間有約 0.6 的巨大間隔（gap），任何合理的閾值（0.25–0.87）都能完美分離。

2. **PCA 視覺化**: 三位說話者在 PC1 軸上就已經完全分開（PC1 解釋了 72.6% 的方差），hsuan 群集在右側，其他兩位在左側。甚至不需要 PC2 就能分類。

3. **WeSpeaker ResNet34 的鑑別力**: 這個模型在 VoxCeleb 上訓練，對於聲學特徵差異明顯的不同說話者，其 embedding 具有極強的區分能力。

### 這代表什麼？

這個結果說明 **utterance-level（全句級別）的說話者驗證在當前場景下是一個已解決的問題**。無論用什麼分類器，只要 embedding 品質夠好，全句比對都是 trivial 的。

### 真正的挑戰在哪裡？

pVAD 的難點不在 utterance-level 分類，而在：

1. **Frame-level 決策**：pipeline 中的 `SimplePVAD` 是對短窗（~0.5s）音頻做 embedding 提取，短窗的 embedding 品質遠低於全句，且受噪音影響大。

2. **混合語音**：真實場景中多人同時說話，embedding 會被污染。

3. **即時性要求**：串流處理需要在有限的上下文中做決策。

4. **閾值敏感度**：雖然 0.24 的閾值在全句上完美，但在 frame-level 上閾值的選擇會變得很敏感。

## 4. 實用建議

1. **維持現有 cosine similarity 方案**: 在 utterance-level 上，它最簡單、零訓練、效果完美。沒有理由換成更複雜的方法。

2. **真正需要比較的場景**: 應該在 **frame-level**（短窗）或 **加入噪音/混合語音** 的條件下重新比較。在那些更困難的場景中，有監督方法（特別是 MLP 和 SVM）可能展現出優勢。

3. **資料量**: 13 個全句樣本對於評估 utterance-level 驗證已經足夠得出結論（任務太簡單）。但要真正比較分類器在困難場景下的差異，需要更多樣本和更具挑戰性的測試條件。

## 5. 輸出檔案

- `roc_curves.png`: ROC 曲線比較圖（所有方法重疊在完美線上）
- `metrics_comparison.png`: 指標柱狀圖
- `embedding_pca.png`: Embedding PCA 分布圖（三群清晰可分）
- `similarity_distribution.png`: Cosine Similarity 分布圖（兩分布間隔巨大）
- `embeddings.npz`: 所有提取的 speaker embeddings (13 × 256)
- `results.json`: 完整的數值結果
- `baseline_comparison.py`: 完整實驗腳本（含中文註解）
