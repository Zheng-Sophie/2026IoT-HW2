"""
pages/5_Feature_Selection.py — 特徵選擇頁（五種方法 + 固定/自動特徵數 + RMSE/R² 曲線）
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FIXED_FEATURE_COUNTS, FEATURE_SELECTION_METHODS, MODEL_NAMES,
    DEFAULT_RANDOM_SEED,
)
from experiment_runner import run_experiments, ExperimentResults
import visualization as viz

st.set_page_config(page_title="Feature Selection | RegressionLab", layout="wide")
st.title("🔬 Feature Selection — 特徵選擇")

if st.session_state.get("df") is None:
    st.info("請先載入資料集。")
    st.stop()

pd_data = st.session_state.get("processed_data")
if pd_data is None:
    st.warning("請先至「Data Preparation」頁面執行資料準備。")
    st.stop()

X_train = pd_data["X_train"]
y_train = pd_data["y_train"]
feat_names = pd_data["feature_names"]
n_available = len(feat_names)

st.markdown(f"""
> **目前可用特徵數：{n_available} 個**
> 特徵選擇在每個 CV Fold 內進行，確保無資料洩漏。
""")

# ── 實驗設定 ──────────────────────────────────────────────────────────────────
st.subheader("1. 實驗設定")

col_a, col_b = st.columns(2)
with col_a:
    selected_methods = st.multiselect(
        "特徵選擇方法", list(FEATURE_SELECTION_METHODS.keys()),
        format_func=lambda k: FEATURE_SELECTION_METHODS[k],
        default=["selectkbest", "rfe"],
        key="exp_methods",
    )
    selected_models = st.multiselect(
        "回歸模型", list(MODEL_NAMES.keys()),
        format_func=lambda k: MODEL_NAMES[k],
        default=["linear"],
        key="exp_models",
    )

with col_b:
    feat_mode = st.radio(
        "特徵數量模式",
        ["fixed", "auto"],
        format_func=lambda x: "固定特徵數比較 (14, 16, 18, 20)" if x == "fixed" else "自動探索最佳特徵數",
        key="feat_mode",
    )

    valid_fixed = [k for k in FIXED_FEATURE_COUNTS if k <= n_available]
    if feat_mode == "fixed":
        if not valid_fixed:
            st.warning(f"可用特徵數 {n_available} 不足，將使用 1~{n_available}。")
        else:
            st.info(f"固定候選特徵數：{valid_fixed}")
    else:
        auto_range = st.slider(
            "自動探索範圍（特徵數）",
            1, n_available, (1, min(n_available, 20)),
            key="auto_range",
        )

# 計算實驗規模
if feat_mode == "fixed":
    n_pts = len(valid_fixed) if valid_fixed else min(n_available, 5)
else:
    lo, hi = auto_range
    n_pts = max(1, (hi - lo) // max(1, (hi - lo) // 10))  # 估計點數
total_combos = len(selected_methods) * n_pts * len(selected_models)
st.caption(f"⚠️ 預計實驗組合數：約 {total_combos} 組（含 {5}-fold CV）。組合數越多耗時越長。")

# ── 執行實驗 ──────────────────────────────────────────────────────────────────
st.subheader("2. 執行特徵選擇實驗")
btn_run = st.button("🚀 開始實驗", type="primary", key="btn_run_fs")

if btn_run:
    if not selected_methods:
        st.error("請至少選擇一種特徵選擇方法。")
    elif not selected_models:
        st.error("請至少選擇一種回歸模型。")
    else:
        progress_bar = st.progress(0, text="準備中...")
        status_text = st.empty()
        results_container = []

        def callback(done, total, msg):
            pct = min(done / max(total, 1), 1.0)
            progress_bar.progress(pct, text=f"[{done}/{total}] {msg}")
            status_text.caption(f"正在執行：{msg}")

        custom_counts = None
        if feat_mode == "auto":
            lo, hi = auto_range
            step = max(1, (hi - lo) // 10)
            custom_counts = list(range(lo, hi + 1, step))
            if hi not in custom_counts:
                custom_counts.append(hi)
            custom_counts = sorted(set(custom_counts))

        with st.spinner("特徵選擇實驗進行中..."):
            results = run_experiments(
                X_train=X_train,
                y_train=y_train,
                method_keys=selected_methods,
                model_keys=selected_models,
                feature_count_mode=feat_mode,
                custom_counts=custom_counts,
                random_seed=st.session_state.get("random_seed", DEFAULT_RANDOM_SEED),
                progress_callback=callback,
            )

        progress_bar.progress(1.0, text="✅ 實驗完成！")
        st.session_state["experiment_results"] = results
        st.success(f"✅ 實驗完成！共 {len(results.points)} 組結果。")

# ── 結果顯示 ──────────────────────────────────────────────────────────────────
results: ExperimentResults = st.session_state.get("experiment_results")
if results and results.points:
    st.subheader("3. RMSE / R² by Number of Features")

    # 控制選項
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        disp_method = st.selectbox(
            "特徵選擇方法", list(FEATURE_SELECTION_METHODS.keys()),
            format_func=lambda k: FEATURE_SELECTION_METHODS[k],
            key="disp_method",
        )
    with ctrl_col2:
        disp_model = st.selectbox(
            "回歸模型", list(MODEL_NAMES.keys()),
            format_func=lambda k: MODEL_NAMES[k],
            key="disp_model",
        )
    with ctrl_col3:
        show_err = st.checkbox("顯示標準差誤差棒", value=True, key="show_err")

    n_list, rmse_m, rmse_s, r2_m, r2_s = results.get_n_features_curve(disp_method, disp_model)
    if n_list:
        suffix = f"{FEATURE_SELECTION_METHODS.get(disp_method, disp_method)} | {MODEL_NAMES.get(disp_model, disp_model)}"
        fig = viz.plot_feature_count_curves(n_list, rmse_m, r2_m, rmse_s, r2_s, suffix, show_err)
        st.pyplot(fig, use_container_width=True)
        st.caption("資料來源：訓練集 Cross-Validation（非 Test Set）")
    else:
        st.info(f"尚未有 {FEATURE_SELECTION_METHODS.get(disp_method)} + {MODEL_NAMES.get(disp_model)} 的實驗結果。請重新執行實驗並選擇對應的方法與模型。")

    # 所有結果表格
    st.subheader("4. 全部實驗結果表")
    df_result = results.as_dataframe()
    st.dataframe(df_result.style.format({
        "CV RMSE (mean)": "{:.6f}", "CV RMSE (std)": "{:.6f}",
        "CV R² (mean)": "{:.6f}", "CV R² (std)": "{:.6f}",
    }), use_container_width=True)

    # 各方法最佳特徵數（按 RMSE 最低）
    st.subheader("5. 各方法推薦特徵數（CV RMSE 最低）")
    best_rows = []
    for mk in selected_methods:
        for mo in selected_models:
            pts = [p for p in results.points if p.method_key == mk and p.model_key == mo]
            if pts:
                best = min(pts, key=lambda p: p.cv_rmse_mean)
                best_rows.append({
                    "特徵選擇方法": FEATURE_SELECTION_METHODS.get(mk, mk),
                    "模型": MODEL_NAMES.get(mo, mo),
                    "推薦特徵數": best.n_features,
                    "CV RMSE": round(best.cv_rmse_mean, 6),
                    "CV R²": round(best.cv_r2_mean, 6),
                    "選中特徵": ", ".join(best.selected_features[:5]) + ("..." if len(best.selected_features) > 5 else ""),
                })
    if best_rows:
        st.dataframe(pd.DataFrame(best_rows), use_container_width=True)
        st.caption("以上推薦基於訓練集 CV 結果，測試集不用於特徵數選擇。")
else:
    st.info("尚未有實驗結果。請設定選項後按「開始實驗」。")
