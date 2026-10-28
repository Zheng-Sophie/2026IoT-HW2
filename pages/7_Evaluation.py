"""
pages/7_Evaluation.py — 模型評估頁（五項指標、診斷、預測區間、模型推薦）
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_SELECTION_METHODS, MODEL_NAMES
from evaluation import compute_metrics, evaluate_on_test, build_comparison_table, recommend_best_model
from diagnostics import run_all_diagnostics
from prediction_intervals import compute_prediction_interval
import visualization as viz

st.set_page_config(page_title="Evaluation | RegressionLab", layout="wide")
st.title("📈 Evaluation — 模型評估")

if st.session_state.get("df") is None:
    st.info("請先載入資料集。")
    st.stop()

pd_data = st.session_state.get("processed_data")
trained = st.session_state.get("trained_models", {})

if pd_data is None:
    st.warning("請先至「Data Preparation」頁面執行資料準備。")
    st.stop()
if not trained:
    st.warning("請先至「Modeling」頁面訓練至少一個模型。")
    st.stop()

X_train = pd_data["X_train"]
X_test  = pd_data["X_test"]
y_train = pd_data["y_train"]
y_test  = pd_data["y_test"]

# ── 選擇評估模型 ──────────────────────────────────────────────────────────────
st.subheader("1. 選擇評估模型")
model_labels = {k: v["label"] for k, v in trained.items()}
sel_key = st.selectbox(
    "選擇已訓練的模型",
    list(model_labels.keys()),
    format_func=lambda k: model_labels[k],
    key="eval_model_sel",
)
eval_btn = st.button("📊 執行評估（含診斷）", type="primary", key="btn_evaluate")

if eval_btn and sel_key:
    model_info = trained[sel_key]
    pipeline = model_info["pipeline"]
    feat_names = model_info["selected_features"]
    model_key = model_info["model_key"]

    with st.spinner("計算測試集指標..."):
        test_eval = evaluate_on_test(pipeline, X_test, y_test, len(feat_names))

    with st.spinner("執行回歸假設診斷..."):
        diag = run_all_diagnostics(
            pipeline, X_train, y_train, X_test, y_test,
            feat_names, model_key
        )

    with st.spinner("計算 95% 預測區間..."):
        pi_info = compute_prediction_interval(
            pipeline, X_train, y_train, X_test[feat_names],
            feat_names, model_key
        )

    st.session_state["evaluation_results"] = {
        "model_key_used": sel_key,
        "test_eval": test_eval,
        "diag": diag,
        "pi_info": pi_info,
        "feat_names": feat_names,
        "model_key": model_key,
        "label": model_info["label"],
    }
    st.session_state["best_model_key"] = sel_key
    st.success("✅ 評估完成！")

# ── 顯示結果 ──────────────────────────────────────────────────────────────────
ev = st.session_state.get("evaluation_results")
if not ev:
    st.info("請選擇模型後按「執行評估」。")
    st.stop()

st.markdown(f"### 評估模型：`{ev['label']}`")
st.caption("以下所有指標均為 **Test Set** 結果（保留集，不用於選模或調參）。")

# ── 五項評估指標 ──────────────────────────────────────────────────────────────
st.subheader("2. 五項評估指標（Test Set）")
m = ev["test_eval"]
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("MAE", f"{m['MAE']:.6f}")
with c2: st.metric("MSE", f"{m['MSE']:.6f}")
with c3: st.metric("RMSE", f"{m['RMSE']:.6f}")
with c4: st.metric("R²", f"{m['R²']:.6f}")
adj_r2_val = m["Adjusted R²"]
with c5:
    if isinstance(adj_r2_val, float):
        st.metric("Adjusted R²", f"{adj_r2_val:.6f}")
    else:
        st.metric("Adjusted R²", str(adj_r2_val))

st.caption("""
**指標說明：** MAE/MSE/RMSE 越低越好；R² 越高越好（可能為負數）；
Adjusted R² 依 n 和 p 計算，Ridge/Lasso 僅作補充比較。
""")

# ── 多模型比較 ────────────────────────────────────────────────────────────────
if len(trained) > 1:
    st.subheader("3. 多模型比較（Test Set）")
    cv_results_for_rec = []
    test_results_for_cmp = []
    for k, v in trained.items():
        try:
            test_res = evaluate_on_test(v["pipeline"], X_test, y_test, v["info"]["n_features"])
            test_results_for_cmp.append({
                "model_label": v["label"],
                "metrics": test_res,
                "source": "Test Set",
            })
            cv_results_for_rec.append({
                "model_label": v["label"],
                "cv_rmse": v["cv_info"]["cv_rmse_mean"],
                "cv_r2": v["cv_info"]["cv_r2_mean"],
                "n_features": v["info"]["n_features"],
            })
        except Exception:
            pass

    if test_results_for_cmp:
        cmp_df = build_comparison_table(test_results_for_cmp)
        st.dataframe(cmp_df.style.format({
            "MAE": "{:.6f}", "MSE": "{:.6f}", "RMSE": "{:.6f}", "R²": "{:.6f}",
        }), use_container_width=True)

        labels = [r["model_label"] for r in test_results_for_cmp]
        rmse_vals = [r["metrics"]["RMSE"] for r in test_results_for_cmp]
        r2_vals = [r["metrics"]["R²"] for r in test_results_for_cmp]
        fig = viz.plot_model_comparison(labels, rmse_vals, r2_vals, "Test Set")
        st.pyplot(fig, use_container_width=True)

    # 模型推薦（基於 CV）
    st.subheader("📌 模型推薦（基於訓練集 CV）")
    best, reasoning = recommend_best_model(cv_results_for_rec)
    if best:
        st.success(f"推薦模型：**{best['model_label']}**")
        st.markdown(reasoning)

# ── 視覺化 ────────────────────────────────────────────────────────────────────
st.subheader("4. Actual vs. Predicted（Test Set）")
y_true = ev["test_eval"]["y_true"]
y_pred = ev["test_eval"]["y_pred"]
residuals = ev["test_eval"]["residuals"]

fig_avp = viz.plot_actual_vs_predicted(y_true, y_pred, f"Actual vs. Predicted — {ev['label']}")
st.pyplot(fig_avp, use_container_width=True)

st.subheader("5. Residuals vs. Fitted（Test Set）")
fig_res = viz.plot_residuals_vs_fitted(y_pred, residuals, f"Residuals vs. Fitted — {ev['label']}")
st.pyplot(fig_res, use_container_width=True)

st.subheader("6. Residual Q-Q Plot（Test Set）")
diag = ev["diag"]
norm_diag = diag.get("normality", {})
if "theoretical_quantiles" in norm_diag:
    fig_qq = viz.plot_qq(norm_diag["theoretical_quantiles"], norm_diag["sample_quantiles"])
    st.pyplot(fig_qq, use_container_width=True)

# ── 五大診斷 ──────────────────────────────────────────────────────────────────
st.subheader("7. 五大回歸假設診斷")
diag_order = [
    ("linearity",         "線性假設（Linearity）"),
    ("independence",      "誤差獨立性（Independence）"),
    ("homoscedasticity",  "等變異性（Homoscedasticity）"),
    ("normality",         "殘差常態性（Normality）"),
    ("multicollinearity", "多重共線性（Multicollinearity）"),
]
for diag_key, diag_title in diag_order:
    d = diag.get(diag_key, {})
    with st.expander(f"**{diag_title}** — {d.get('applicability', '')}"):
        st.markdown(f"**方法：** {d.get('method', '—')}")
        st.markdown(f"**適用性：** {d.get('applicability', '—')}")
        interp = d.get("interpretation", "")
        if interp: st.info(interp)
        note = d.get("note", "")
        if note: st.caption(note)

        if diag_key == "independence" and d.get("dw_statistic") is not None:
            st.metric("Durbin–Watson 統計量", f"{d['dw_statistic']:.4f}")
        if diag_key == "homoscedasticity":
            if d.get("bp_statistic") is not None:
                c1, c2 = st.columns(2)
                with c1: st.metric("Breusch–Pagan 統計量", f"{d['bp_statistic']:.4f}")
                with c2: st.metric("p-value", f"{d['bp_pvalue']:.4f}")
        if diag_key == "normality":
            if d.get("sw_statistic") is not None:
                c1, c2 = st.columns(2)
                with c1: st.metric("Shapiro–Wilk 統計量", f"{d['sw_statistic']:.4f}")
                with c2: st.metric("p-value", f"{d['sw_pvalue']:.4f}")
            if d.get("large_sample_warning"):
                st.warning("⚠️ 大樣本警告：Shapiro–Wilk在大樣本下幾乎必然顯著，建議觀察Q-Q Plot。")
        if diag_key == "multicollinearity" and d.get("vif_table") is not None:
            st.markdown("**VIF 表**")
            st.dataframe(d["vif_table"], use_container_width=True)

# ── 95% Prediction Interval ───────────────────────────────────────────────────
st.subheader("8. 95% 預測區間（Prediction Interval）")
pi_info = ev.get("pi_info", {})
if pi_info.get("supported"):
    st.success("✅ 已計算 OLS 95% 預測區間（個別觀測）")
    st.markdown(f"**方法：** {pi_info.get('method', '—')}")
    st.info(pi_info.get("note", ""))
    # 顯示前20個測試樣本的區間
    pi_df = pd.DataFrame({
        "Actual": y_true[:20],
        "Predicted": pi_info["y_pred"][:20],
        "PI Lower (95%)": pi_info["pi_lower"][:20],
        "PI Upper (95%)": pi_info["pi_upper"][:20],
    }).round(6)
    st.dataframe(pi_df, use_container_width=True)
    st.caption("僅顯示前 20 筆測試集樣本。95% PI 基於 OLS 的誤差常態性與等變異性假設。")
else:
    st.warning(f"⚠️ {pi_info.get('note', '此模型不支援直接計算 95% 預測區間。')}")
