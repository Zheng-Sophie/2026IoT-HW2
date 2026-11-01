"""
pages/6_Modeling.py — 模型建置頁（Linear/Ridge/Lasso + 特徵選擇 + 訓練）
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_SELECTION_METHODS, MODEL_NAMES, DEFAULT_RANDOM_SEED, CV_FOLDS,
)
from feature_selection import run_feature_selection
from modeling import fit_model, cv_evaluate_model
from evaluation import compute_metrics
import visualization as viz

st.set_page_config(page_title="Modeling | RegressionLab", layout="wide")
st.title("🤖 Modeling — 模型建置")

if st.session_state.get("df") is None:
    st.info("請先載入資料集。")
    st.stop()

pd_data = st.session_state.get("processed_data")
if pd_data is None:
    st.warning("請先至「Data Preparation」頁面執行資料準備。")
    st.stop()

X_train = pd_data["X_train"]
X_test  = pd_data["X_test"]
y_train = pd_data["y_train"]
y_test  = pd_data["y_test"]
feat_names = pd_data["feature_names"]
n_available = len(feat_names)
seed = st.session_state.get("random_seed", DEFAULT_RANDOM_SEED)

st.markdown("""
> 在此頁面選擇特徵選擇方法、特徵數量與回歸模型，然後訓練模型。
> Ridge/Lasso 的 alpha 使用訓練集 CV 自動選取；測試集保留至評估頁面。
""")

# ── 訓練設定 ──────────────────────────────────────────────────────────────────
st.subheader("1. 模型設定")
col1, col2, col3 = st.columns(3)
with col1:
    method_key = st.selectbox(
        "特徵選擇方法", list(FEATURE_SELECTION_METHODS.keys()),
        format_func=lambda k: FEATURE_SELECTION_METHODS[k],
        key="model_method",
    )
with col2:
    n_features = st.slider("特徵數量", 1, n_available, min(16, n_available), key="model_n_feat")
with col3:
    model_key = st.selectbox(
        "回歸模型", list(MODEL_NAMES.keys()),
        format_func=lambda k: MODEL_NAMES[k],
        key="model_type",
    )

st.subheader("2. 訓練")
btn_train = st.button("🚀 訓練模型", type="primary", key="btn_train_model")

if btn_train:
    with st.spinner(f"特徵選擇中（{FEATURE_SELECTION_METHODS[method_key]}，k={n_features}）..."):
        sel_result = run_feature_selection(method_key, X_train, y_train, n_features, seed)
    selected_feats = sel_result.selected_features
    st.info(f"選中特徵（{len(selected_feats)}）：{', '.join(selected_feats)}")

    X_train_selected = X_train[selected_feats].copy()

    with st.spinner(f"訓練 {MODEL_NAMES[model_key]} ..."):
        pipeline, info = fit_model(model_key, selected_feats, X_train_selected, y_train, seed)
        cv_info = cv_evaluate_model(model_key, selected_feats, X_train_selected, y_train, CV_FOLDS, seed)

    model_label = f"{MODEL_NAMES[model_key]} | {FEATURE_SELECTION_METHODS[method_key]} | k={len(selected_feats)}"
    model_state_key = f"{model_key}_{method_key}_{len(selected_feats)}"

    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    st.session_state["trained_models"][model_state_key] = {
        "pipeline": pipeline,
        "info": info,
        "cv_info": cv_info,
        "label": model_label,
        "selection_result": sel_result,
        "method_key": method_key,
        "model_key": model_key,
        "selected_features": selected_feats,
    }
    # 更新最新模型作為預測 pipeline
    st.session_state["prediction_pipeline"] = {
        "pipeline": pipeline,
        "feature_names": selected_feats,
        "model_type": model_key,
        "model_label": model_label,
        "X_train": X_train_selected,
        "y_train": y_train,
    }
    st.success(f"✅ 訓練完成！CV RMSE={cv_info['cv_rmse_mean']:.6f}，CV R²={cv_info['cv_r2_mean']:.6f}")

# ── 顯示已訓練的模型 ──────────────────────────────────────────────────────────
trained = st.session_state.get("trained_models", {})
if trained:
    st.subheader("3. 已訓練模型摘要")
    summary_rows = []
    for k, v in trained.items():
        info = v["info"]
        cv_info = v["cv_info"]
        row = {
            "模型標籤": v["label"],
            "特徵選擇方法": FEATURE_SELECTION_METHODS.get(v["method_key"], v["method_key"]),
            "模型類型": MODEL_NAMES.get(v["model_key"], v["model_key"]),
            "特徵數": info["n_features"],
            "CV RMSE": round(cv_info["cv_rmse_mean"], 6),
            "CV RMSE std": round(cv_info["cv_rmse_std"], 6),
            "CV R²": round(cv_info["cv_r2_mean"], 6),
        }
        if "best_alpha" in info:
            row["Best Alpha"] = info["best_alpha"]
        if "non_zero_features" in info:
            row["Non-zero Coefs"] = info["non_zero_features"]
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True)

    # 選取查看模型
    st.subheader("4. 查看模型係數")
    model_labels = {k: v["label"] for k, v in trained.items()}
    sel_key = st.selectbox("選擇查看的模型", list(model_labels.keys()),
                           format_func=lambda k: model_labels[k], key="view_model_sel")
    if sel_key and sel_key in trained:
        v = trained[sel_key]
        info = v["info"]
        coefs = info.get("coefficients", {})
        if coefs:
            fig = viz.plot_coefficients(
                list(coefs.keys()), list(coefs.values()),
                title=f"係數圖 — {v['label']}",
            )
            st.pyplot(fig, use_container_width=True)
        st.markdown("**係數詳細（排序：絕對值大到小）**")
        coef_df = pd.DataFrame(coefs.items(), columns=["特徵", "係數"])
        coef_df["絕對值"] = coef_df["係數"].abs()
        coef_df = coef_df.sort_values("絕對值", ascending=False).drop("絕對值", axis=1)
        st.dataframe(coef_df.style.format({"係數": "{:.6f}"}), use_container_width=True)

    st.caption("""
    ⚠️ CV 分數用於模型選擇參考。若比較多個模型並從中挑選，CV 分數存在樂觀偏差。
    最終公正的泛化效能估計請至「Evaluation」頁面查看 Test Set 結果。
    """)
