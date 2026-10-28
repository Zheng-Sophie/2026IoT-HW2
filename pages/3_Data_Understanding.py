"""
pages/3_Data_Understanding.py — 資料理解頁
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_validation import validate_dataframe, describe_numeric
import visualization as viz

st.set_page_config(page_title="Data Understanding | RegressionLab", layout="wide")
st.title("🔍 Data Understanding — 資料理解")

if st.session_state.get("df") is None:
    st.info("請先載入資料集。")
    st.stop()

df = st.session_state.df
target = st.session_state.get("target_col", "")
feat_cols = st.session_state.get("feature_cols", [])
report = validate_dataframe(df)

# ── 1. 基本資訊 ───────────────────────────────────────────────────────────────
st.subheader("1. 基本資訊")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("資料列數", f"{report['n_rows']:,}")
with c2: st.metric("欄位數", report["n_cols"])
with c3: st.metric("目標欄位", target)
with c4: st.metric("候選特徵數", len(feat_cols))

c5, c6, c7, c8 = st.columns(4)
with c5: st.metric("數值欄位", len(report["numeric_cols"]))
with c6: st.metric("非數值欄位", len(report["non_numeric_cols"]))
with c7: st.metric("有缺失值的欄位", len(report["missing"]))
with c8: st.metric("重複列", report["duplicate_rows"])

# ── 2. 欄位型態 ───────────────────────────────────────────────────────────────
st.subheader("2. 欄位型態")
dtype_df = pd.DataFrame({"欄位": df.columns, "型態": df.dtypes.astype(str),
    "唯一值": [df[c].nunique() for c in df.columns],
    "缺失值": [df[c].isnull().sum() for c in df.columns],
    "缺失比例(%)": [(df[c].isnull().sum()/len(df)*100).round(2) for c in df.columns]})
st.dataframe(dtype_df, use_container_width=True)

# ── 3. 常數欄位 & 可能 ID 欄位 ────────────────────────────────────────────────
if report["constant_cols"] or report["likely_id_cols"]:
    st.subheader("3. 特殊欄位")
    if report["constant_cols"]:
        st.warning(f"⚠️ 常數欄位（無變異量，建議排除）：`{'`, `'.join(report['constant_cols'])}`")
    if report["likely_id_cols"]:
        st.warning(f"⚠️ 可能為 ID 欄位（建議排除）：`{'`, `'.join(report['likely_id_cols'])}`")

# ── 4. 描述統計 ───────────────────────────────────────────────────────────────
st.subheader("4. 數值欄位描述統計")
if report["numeric_cols"]:
    desc = describe_numeric(df, report["numeric_cols"])
    st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

# ── 5. 目標變數分布 ───────────────────────────────────────────────────────────
st.subheader("5. 目標變數分布")
if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
    fig = viz.plot_target_distribution(df[target], target)
    st.pyplot(fig, use_container_width=True)
else:
    st.warning(f"目標欄位 `{target}` 不可用於繪圖。")

# ── 6. 相關性熱圖 ─────────────────────────────────────────────────────────────
st.subheader("6. 特徵相關性熱圖")
cols_for_corr = [c for c in feat_cols + ([target] if target else []) if c in df.columns]
if len(cols_for_corr) >= 2:
    corr = df[cols_for_corr].corr()
    fig = viz.plot_correlation_heatmap(corr, f"Correlation Heatmap（含目標 {target}）")
    st.pyplot(fig, use_container_width=True)

    st.markdown("**與目標變數相關係數（Top 15）**")
    if target in corr.columns:
        corr_target = corr[target].drop(target, errors="ignore").abs().sort_values(ascending=False).head(15)
        st.dataframe(corr_target.reset_index().rename(columns={"index": "特徵", target: "| 相關係數 |"}), use_container_width=True)
else:
    st.info("特徵數量不足，無法繪製相關性熱圖。")

# ── 7. 可能的異常值 ───────────────────────────────────────────────────────────
st.subheader("7. 可能的異常值（IQR 方法）")
if feat_cols:
    outlier_counts = {}
    for c in feat_cols[:20]:  # 限制前20個特徵
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            Q1 = df[c].quantile(0.25)
            Q3 = df[c].quantile(0.75)
            IQR = Q3 - Q1
            out = ((df[c] < Q1 - 1.5 * IQR) | (df[c] > Q3 + 1.5 * IQR)).sum()
            if out > 0:
                outlier_counts[c] = out
    if outlier_counts:
        out_df = pd.DataFrame(outlier_counts.items(), columns=["特徵", "疑似異常值數量"])
        out_df["比例(%)"] = (out_df["疑似異常值數量"] / len(df) * 100).round(2)
        st.dataframe(out_df.sort_values("疑似異常值數量", ascending=False), use_container_width=True)
    else:
        st.success("以 IQR 方法檢查，未發現明顯異常值。")

# ── 8. 資料品質摘要 ───────────────────────────────────────────────────────────
if report["warnings"]:
    st.subheader("8. 資料品質警告")
    for w in report["warnings"]:
        st.warning(w)
else:
    st.success("✅ 資料品質初步檢查通過，無明顯問題。")
