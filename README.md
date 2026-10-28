# RegressionLab

互動式多元線性回歸分析平台，遵循 CRISP-DM 六階段流程。

## 快速啟動

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 啟動應用
streamlit run app.py
```

瀏覽器會自動開啟 `http://localhost:8501`

---

## 資料集設定

### 預設資料集：Predict the Flood（Kaggle）

本應用預設使用 Kaggle 競賽資料集：
- **競賽頁面**：https://www.kaggle.com/competitions/predict-the-flood/data
- **目標欄位**：`FloodProbability`

**取得資料步驟：**
1. 登入 Kaggle 帳號並同意競賽規則
2. 下載 `train.csv`
3. 將 `train.csv` 放置於本專案目錄（與 `app.py` 同一層）
4. 重新整理 Streamlit 頁面，點擊「載入預設資料集」

### 備用方式
- 在「Dataset Source」面板使用「上傳 CSV」功能直接上傳本機檔案
- 輸入可直接存取的公開 CSV URL（如 raw GitHub URL）

---

## 功能說明

| 頁面 | 功能 |
|---|---|
| Overview | 專案狀態、資料集摘要、分析進度 |
| Business Understanding | 研究目標、問題定義、成功標準 |
| Data Understanding | 描述統計、缺失值、相關性熱圖、異常值 |
| Data Preparation | 抽樣（5k/20k/50k/全部）、80/20 切分、前處理 Pipeline |
| Feature Selection | 五種方法、固定/自動特徵數、RMSE/R² 折線圖 |
| Modeling | 三種回歸模型、CV 調參、係數圖 |
| Evaluation | 五項指標、五大診斷、Actual vs. Predicted、95% PI |
| Deployment | 單筆預測、CSV 批次預測 |

---

## 五種特徵選擇方法

1. **SelectKBest** — f_regression 分數
2. **RFE** — Recursive Feature Elimination
3. **SFS** — Sequential Feature Selection（前向）
4. **Lasso Feature Selection** — LassoCV 非零係數
5. **Permutation Importance** — 驗證折 importance 估計

---

## 三種回歸模型

| 模型 | Alpha 選取 |
|---|---|
| Linear Regression | 無（OLS 基準） |
| Ridge Regression | 訓練集 CV 自動選取 |
| Lasso Regression | 訓練集 CV 自動選取 |

---

## 已知限制

- **Kaggle 資料集** 需自行下載（需登入），無法自動取得
- **SFS** 計算較慢，建議使用快速模式（3-fold 內部 CV）
- **Ridge/Lasso 預測區間** 第一版不支援（標示為「尚未支援」）
- **Shapiro–Wilk** 大樣本（>5,000）會幾乎必然顯著，建議觀察 Q-Q Plot
- **第一版不提供** PDF/Excel/CSV 分析報告或預測結果匯出

---

## 套件版本

詳見 `requirements.txt`。主要依賴：
- `streamlit>=1.32.0`
- `scikit-learn>=1.4.0`
- `statsmodels>=0.14.0`
- `pandas>=2.0.0`
- `matplotlib>=3.8.0`

---

## 統計正確性聲明

- 所有圖表與指標來自真實分析結果，無假數據或硬編碼示例分數
- 特徵選擇器放在 CV Pipeline 內，防止洩漏
- 測試集僅用於最終評估，不參與選模或調參
- Ridge/Lasso 不套用 OLS 預測區間公式

> **資料集聲明：** Predict the Flood 為合成資料，結果不代表真實洪災預測能力。
