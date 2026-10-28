"""
pages/2_Business_Understanding.py — 商業理解頁
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Business Understanding | RegressionLab", layout="wide")
st.title("💼 Business Understanding — 商業理解")

st.markdown("""
## 1. 研究目的

本專案使用 **多元線性回歸（Multiple Linear Regression）** 方法，探討：

> **「有哪些氣候與環境特徵能夠預測洪水發生機率（FloodProbability）？
> 不同特徵數量如何影響模型的預測效能？」**

---

## 2. 專案背景

本專案以 Kaggle 競賽 **Predict the Flood** 資料集為主要研究對象。
資料集包含多個可能影響洪水發生機率的環境與氣候特徵。

### 重要聲明

> ⚠️ **本資料集為合成（Synthetic）資料**，研究結果**不能**直接視為真實洪災預警能力的證明。
> 模型效能指標（RMSE、R²）反映的是在此合成資料集上的擬合程度，
> 而非對真實世界洪水事件預測的保證。

---

## 3. CRISP-DM 流程

本專案遵循 **CRISP-DM（Cross-Industry Standard Process for Data Mining）** 六階段：

| 階段 | 內容 |
|---|---|
| Business Understanding | 定義研究目標、成功標準 |
| Data Understanding | 探索資料結構、品質與分布 |
| Data Preparation | 抽樣、切分、前處理 |
| Feature Selection | 五種方法選取最佳特徵子集 |
| Modeling | Linear、Ridge、Lasso 回歸比較 |
| Evaluation | 五項指標、模型診斷、推薦模型 |

---

## 4. 模型成功標準

模型選擇依據以下多準則：

1. **RMSE**（均方根誤差）：越低越好，與目標變數同單位。
2. **R²**（決定係數）：越高越好，衡量模型相對基準的解釋能力。
3. **模型解釋性**：係數可解釋性與特徵數量。

> ℹ️ 最終推薦模型基於訓練集 Cross-Validation 結果。
> 測試集結果用於無偏的最終效能估計，不用於選模或調參。

---

## 5. 研究問題定義

- **預測目標**：`FloodProbability`（連續數值，0 ~ 1）
- **問題類型**：迴歸（Regression）
- **分析主題**：不同特徵數量對 Linear / Ridge / Lasso 回歸效能的影響
- **不在範圍內**：分類問題、深度學習、神經網路、時間序列預測
""")

if st.session_state.get("df") is not None:
    df = st.session_state.df
    target = st.session_state.get("target_col", "FloodProbability")
    st.markdown("---")
    st.subheader("📋 目前載入資料集的基本資訊")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("資料集名稱", st.session_state.get("dataset_name", "—"))
    with col2:
        st.metric("資料列數", f"{len(df):,}")
    with col3:
        st.metric("目標欄位", target)
    if target in df.columns:
        st.write(f"目標欄位 `{target}` 統計摘要：")
        st.dataframe(df[[target]].describe().T.style.format("{:.6f}"))
