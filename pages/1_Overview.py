"""
pages/1_Overview.py — 總覽頁
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_validation import validate_dataframe

st.set_page_config(page_title="Overview | RegressionLab", layout="wide")
st.title("🏠 Overview — 專案總覽")

if st.session_state.get("df") is None:
    st.info("請先從首頁（app.py）的 Dataset Source 面板載入資料集。")
    st.stop()

df = st.session_state.df
target = st.session_state.get("target_col", "")
feat_cols = st.session_state.get("feature_cols", [])

st.markdown("""
## 關於 RegressionLab

RegressionLab 是以 **CRISP-DM** 六階段為框架的多元線性回歸互動分析平台，
主題為「不同特徵數量如何影響回歸模型效能」。

支援三種回歸模型（Linear、Ridge、Lasso）、五種特徵選擇方法，
並提供完整的統計診斷與預測功能。
""")

st.markdown("---")

# ── 目前資料集狀態 ────────────────────────────────────────────────────────────
st.subheader("📁 目前資料集")
report = validate_dataframe(df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("資料集名稱", st.session_state.get("dataset_name", "—"))
with col2:
    st.metric("資料列數", f"{report['n_rows']:,}")
with col3:
    st.metric("欄位數", report["n_cols"])
with col4:
    st.metric("候選特徵數", len(feat_cols))

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric("目標欄位", target)
with col6:
    st.metric("數值欄位", len(report["numeric_cols"]))
with col7:
    st.metric("缺失值欄位", len(report["missing"]))
with col8:
    st.metric("重複列", report["duplicate_rows"])

st.markdown("---")

# ── 實驗設定 ──────────────────────────────────────────────────────────────────
st.subheader("⚙️ 目前實驗設定")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("抽樣數", str(st.session_state.get("sample_size", "—")))
with col_b:
    st.metric("訓練/測試比", f"{st.session_state.get('train_ratio', 0.8)*100:.0f}% / {(1-st.session_state.get('train_ratio', 0.8))*100:.0f}%")
with col_c:
    st.metric("隨機種子", st.session_state.get("random_seed", 42))

st.markdown("---")

# ── 分析進度 ──────────────────────────────────────────────────────────────────
st.subheader("📊 分析進度")

steps = [
    ("Business Understanding", "✅ 已就緒", True),
    ("Data Understanding", "✅ 已就緒" if df is not None else "⏳ 等待載入", df is not None),
    ("Data Preparation", "✅ 已完成" if st.session_state.get("processed_data") else "⏳ 尚未執行", bool(st.session_state.get("processed_data"))),
    ("Feature Selection", "✅ 已完成" if st.session_state.get("experiment_results") else "⏳ 尚未執行", bool(st.session_state.get("experiment_results"))),
    ("Modeling", "✅ 已訓練" if st.session_state.get("trained_models") else "⏳ 尚未訓練", bool(st.session_state.get("trained_models"))),
    ("Evaluation", "✅ 已完成" if st.session_state.get("evaluation_results") else "⏳ 尚未執行", bool(st.session_state.get("evaluation_results"))),
    ("Deployment", "✅ 就緒" if st.session_state.get("prediction_pipeline") else "⏳ 等待模型", bool(st.session_state.get("prediction_pipeline"))),
]

for name, status, done in steps:
    icon = "✅" if done else "⏳"
    st.markdown(f"- {icon} **{name}** — {status}")

if report.get("warnings"):
    st.markdown("---")
    st.subheader("⚠️ 資料品質警告")
    for w in report["warnings"]:
        st.warning(w)
