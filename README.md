# RegressionLab – 多元線性迴歸分析平台

一個以 Python 與 Streamlit 開發的互動式多元線性迴歸分析平台，整合資料前處理、特徵選擇、模型訓練、效能評估及預測功能，並依循 CRISP-DM（Cross-Industry Standard Process for Data Mining）資料探勘標準流程，提供完整的回歸分析實驗環境。

---

Website：https://agnes-iot-hw2.streamlit.app/
<img width="1350" height="721" alt="image" src="https://github.com/user-attachments/assets/bd66eed1-12ba-4d3d-89b0-98d97812bf9e" />


## 1. 專案介紹（Project Overview）

RegressionLab 是一個以多元線性迴歸（Multiple Linear Regression, MLR）為核心的資料分析平台。

本專案透過互動式介面，讓使用者能夠載入不同的 CSV 資料集，自動進行資料檢查與前處理，並比較不同特徵選擇方法及迴歸模型的預測效果。

平台除了提供單一模型的訓練結果，也能透過不同特徵數量的實驗，觀察特徵數量與模型預測效能之間的關係，進一步找出適合的特徵組合。

**專案主要目標：**
- 建立完整的多元線性迴歸分析流程。
- 整合多種特徵選擇方法，比較不同特徵組合的效果。
- 比較不同迴歸模型的預測能力。
- 透過交叉驗證尋找適合的模型參數。
- 提供視覺化分析及模型診斷。
- 支援單筆資料與 CSV 批次預測。

## 2. 使用資料集（Dataset）

### 預設資料集：Student Performance Factors

本專案預設使用 `Student Performance Factors.csv` 作為分析資料集，研究學生學習相關因素與考試成績之間的關係。

使用者也可以透過平台提供的功能載入其他 CSV 資料集，進行不同資料集的回歸分析。

**資料來源：**
- Kaggle – Student Performance Factors
- Dataset 格式：CSV

**資料載入方式：**
1. 系統啟動時自動載入預設資料集。
2. 使用者可以上傳自己的 CSV 檔案。
3. 使用者可以透過 Dataset URL 載入外部資料集。

系統會針對載入的資料進行基本驗證，包含資料型態、缺失值、欄位資訊及目標變數等檢查，以確保資料符合回歸分析需求。

## 3. 系統功能（System Features）

### 3.1 Dataset Management
- 自動載入預設 CSV 資料集。
- 支援使用者上傳 CSV。
- 支援透過 Dataset URL 載入資料。
- 自動檢查資料筆數、欄位數量及資料型態。
- 檢查缺失值及資料品質。
- 支援數值型與類別型特徵的前處理。

### 3.2 Data Understanding
- 資料集基本資訊。
- 特徵欄位與目標變數分析。
- 敘述性統計（Descriptive Statistics）。
- 缺失值分布分析。
- 特徵相關係數分析。
- 相關係數熱圖（Correlation Heatmap）。

### 3.3 Feature Selection（特徵選擇）

平台整合五種特徵選擇方法，用於評估不同特徵組合對模型預測效能的影響。

| 方法 | 全名 | 說明 |
|---|---|---|
| SelectKBest | Univariate Feature Selection | 根據單一特徵的統計分數，選出排名前 K 個特徵 |
| RFE | Recursive Feature Elimination | 透過遞迴方式移除重要性較低的特徵 |
| SFS | Sequential Feature Selection | 透過逐步加入或移除特徵，尋找較佳特徵組合 |
| Lasso Feature Selection | L1-based Feature Selection | 利用 L1 正則化使部分特徵係數縮減至零 |
| Permutation Importance | Permutation Feature Importance | 透過打亂特徵值，觀察模型預測效能下降程度，評估特徵重要性 |

**分析重點：**
- 比較五種特徵選擇方法的結果。
- 調整不同特徵數量（Number of Features）。
- 分析特徵數量與模型預測效能之間的關係。
- 找出適合的特徵組合與最佳特徵數量。

透過視覺化圖表，以特徵數量為 X 軸，比較不同特徵組合的 RMSE 與 R²，觀察特徵選擇對模型效能的影響。

### 3.4 Regression Models（迴歸模型）

平台整合三種迴歸模型：

| Model | 說明 |
|---|---|
| Linear Regression | 基本多元線性迴歸模型 |
| Ridge Regression | 使用 L2 正則化，降低模型係數過大的問題 |
| Lasso Regression | 使用 L1 正則化，可進行係數收縮及特徵篩選 |

模型使用訓練資料進行交叉驗證（Cross-Validation），調整相關超參數，並以測試資料評估模型的泛化能力。

### 3.5 Model Evaluation（模型評估）

採用 80% Training / 20% Testing 的資料切分方式，透過多種評估指標比較模型的預測能力。

**評估指標（Evaluation Metrics）：**

| 指標 | 全名 | 評估方向 | 說明 |
|---|---|---|---|
| MAE | Mean Absolute Error | 越低越好 | 預測值與實際值的平均絕對誤差 |
| MSE | Mean Squared Error | 越低越好 | 預測誤差平方的平均值 |
| RMSE | Root Mean Squared Error | 越低越好 | MSE 的平方根，與目標變數具有相同單位 |
| R² | Coefficient of Determination | 越高越好 | 衡量模型解釋目標變數變異的能力 |
| Adjusted R² | Adjusted Coefficient of Determination | 越高越好 | 考量模型複雜度後調整的決定係數 |

**評估指標說明：**
- MAE、MSE、RMSE：數值越低代表預測誤差越小。
- R²：數值越高代表模型對目標變數變異的解釋能力越高。
- R² 可能為負數，表示模型表現甚至不如直接使用目標變數平均值作為預測。
- Adjusted R²：用於補充評估模型的解釋能力。

此外，平台提供迴歸模型診斷功能，檢查模型的統計假設及適用性，並提供 95% Prediction Interval，呈現個別預測結果的不確定性。

### 3.6 Prediction（預測）

提供兩種預測方式：

**Single Prediction**
- 使用者輸入單筆特徵資料。
- 透過訓練完成的模型產生預測結果。

**Batch Prediction**
- 上傳 CSV 檔案。
- 對多筆資料進行批次預測。
- 顯示各筆資料的預測結果。

### 3.7 Prediction Interval（預測區間）

提供 95% Prediction Interval（預測區間），呈現模型對個別新觀測值的預測不確定性。

同時說明 Confidence Interval（信賴區間）與 Prediction Interval（預測區間）的差異，協助使用者理解模型預測結果。

---

## 4. CRISP-DM 分析流程

本專案依循 CRISP-DM 六個階段進行資料探勘與模型開發。

| 階段 | 名稱 | 實作內容 |
|---|---|---|
| 1 | Business Understanding | 確認分析目標及研究問題 |
| 2 | Data Understanding | 載入資料、統計分析、相關性分析 |
| 3 | Data Preparation | 缺失值處理、類別編碼、特徵前處理 |
| 4 | Modeling | 特徵選擇、模型訓練、交叉驗證 |
| 5 | Evaluation | 模型效能比較、模型診斷及預測區間 |
| 6 | Deployment | 透過 Streamlit 提供互動式分析及預測介面 |

### CRISP-DM Workflow

```text
Business Understanding
          |
          v
  Data Understanding
          |
          v
  Data Preparation
          |
          v
      Modeling
          |
          v
     Evaluation
          |
          v
     Deployment
```

---

## 5. 使用技術（Technology Stack）

| 技術 | 用途 |
|---|---|
| Python | 主要開發語言 |
| Streamlit | 網頁互動介面 |
| Pandas | 資料讀取與處理 |
| NumPy | 數值運算 |
| Scikit-learn | 特徵選擇、模型訓練及交叉驗證 |
| Matplotlib | 資料視覺化 |
| Seaborn | 統計圖表及相關係數熱圖 |

---

## 6. 系統架構（Project Structure）

以下為專案目錄示意，實際檔案名稱請依照 Repository 內容調整。

```text
RegressionLab/
│
├── app.py                  # Streamlit 主程式
│
├── pages/                  # 系統各功能頁面
│   ├── 1_*.py
│   ├── 2_*.py
│   ├── 3_*.py
│   └── ...
│
├── data/
│   └── Student Performance Factors.csv
│
├── requirements.txt        # Python 套件需求
│
├── .gitignore              # Git 忽略設定
│
└── README.md               # 專案說明文件
```

---

## 7. 安裝與執行（Installation & Usage）

### 7.1 環境需求

- Python 3.10 以上
- pip

### 7.2 安裝套件

```bash
pip install -r requirements.txt
```

### 7.3 啟動系統

```bash
streamlit run app.py
```

執行後，Streamlit 會啟動本機網頁伺服器，使用瀏覽器即可進入 RegressionLab 分析平台。

---

## 8. 部署（Deployment）

本專案使用 Streamlit Community Cloud 進行線上部署。

部署步驟：

1. 將專案程式碼上傳至 GitHub Repository。
2. 確認 `requirements.txt` 包含所需套件。
3. 確認預設 CSV 已放置於 `data/` 資料夾。
4. 至 Streamlit Community Cloud 建立新應用程式。
5. 選擇 GitHub Repository 及 `app.py` 作為啟動入口。
6. 完成部署後即可透過公開網址使用系統。

---

## 9. 專案特色（Highlights）

- **Interactive Analysis：** 透過 Streamlit 提供互動式資料分析介面。
- **Multiple Feature Selection：** 比較五種特徵選擇方法。
- **Model Comparison：** 比較 Linear Regression、Ridge Regression 與 Lasso Regression。
- **Cross Validation：** 透過交叉驗證進行模型參數調整。
- **Model Diagnostics：** 提供迴歸模型診斷及假設檢查。
- **Prediction Interval：** 使用 95% 預測區間呈現預測不確定性。
- **Flexible Dataset：** 支援預設資料集、CSV 上傳及外部資料來源。
- **CRISP-DM：** 完整整合資料探勘六階段流程。

---

## 10. Learning Outcomes

透過本次作業，實際整合資料探勘及機器學習的相關技術，從資料取得、資料清理、特徵選擇到模型訓練與評估，完成一套完整的多元線性迴歸分析平台。

本次實作的主要學習成果：

1. 熟悉 CRISP-DM 資料探勘流程。
2. 理解資料前處理對模型訓練的影響。
3. 比較不同特徵選擇方法的優缺點。
4. 理解 Linear Regression、Ridge 與 Lasso 的差異。
5. 學習使用 Cross-Validation 進行模型參數調整。
6. 透過 MAE、MSE、RMSE、R² 及 Adjusted R² 等指標評估模型預測能力。
7. 理解迴歸模型診斷與預測區間。
8. 運用 AI Coding 輔助開發，將機器學習演算法整合成可操作的網頁應用程式。

---

**Project:** RegressionLab – Multiple Linear Regression Analysis Platform  
**Framework:** Python / Streamlit  
**Methodology:** CRISP-DM  
**Models:** Linear Regression / Ridge / Lasso
