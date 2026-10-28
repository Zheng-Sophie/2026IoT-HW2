"""
diagnostics.py — 五大回歸假設診斷（含適用性標籤）

假設與方法:
  1. Linearity      → Residuals vs. Fitted plot
  2. Independence   → Durbin–Watson
  3. Homoscedasticity → Residual spread + Breusch–Pagan
  4. Normality      → Q-Q plot + Shapiro–Wilk
  5. Multicollinearity → VIF + Correlation matrix
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


APPLICABILITY_LABELS = {
    "applicable":   "✅ 適用",
    "supplemental": "⚠️ 補充診斷",
    "conditional":  "ℹ️ 需依資料結構判斷",
    "na":           "❌ 不適用",
}


def _label(model_type: str, diag: str) -> str:
    """根據模型類型，回傳適用性標籤。"""
    if model_type == "linear":
        return APPLICABILITY_LABELS["applicable"]
    # Ridge / Lasso
    if diag in ("homoscedasticity", "normality"):
        return APPLICABILITY_LABELS["supplemental"]
    if diag == "independence":
        return APPLICABILITY_LABELS["conditional"]
    return APPLICABILITY_LABELS["applicable"]


# ── 1. 線性假設 ────────────────────────────────────────────────────────────────
def check_linearity(
    y_fitted: np.ndarray,
    residuals: np.ndarray,
    model_type: str = "linear",
) -> dict:
    """Residuals vs. Fitted 視覺化診斷（回傳資料，繪圖在 visualization.py）。"""
    return {
        "name": "線性假設（Linearity）",
        "method": "Residuals vs. Fitted Plot",
        "y_fitted": y_fitted,
        "residuals": residuals,
        "applicability": _label(model_type, "linearity"),
        "interpretation": (
            "若殘差呈隨機分布，無明顯型態，則線性假設基本成立。"
            "若出現曲線、漏斗或其他系統性型態，表示可能有非線性關係或重要特徵遺漏。"
        ),
        "note": "無法單靠此圖做絕對判定，建議結合其他診斷。",
    }


# ── 2. 獨立性假設 ──────────────────────────────────────────────────────────────
def check_independence(
    residuals: np.ndarray,
    model_type: str = "linear",
) -> dict:
    """Durbin–Watson 統計量（2 附近表示不自相關）。"""
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = float(durbin_watson(residuals))
        if dw_stat < 1.5:
            interp = f"DW={dw_stat:.4f}：可能存在正自相關（DW < 1.5）。"
        elif dw_stat > 2.5:
            interp = f"DW={dw_stat:.4f}：可能存在負自相關（DW > 2.5）。"
        else:
            interp = f"DW={dw_stat:.4f}：殘差大致無明顯自相關（1.5 < DW < 2.5）。"
    except Exception as e:
        dw_stat = None
        interp = f"無法計算 Durbin–Watson：{e}"

    return {
        "name": "誤差獨立性（Independence）",
        "method": "Durbin–Watson 統計量",
        "dw_statistic": dw_stat,
        "applicability": _label(model_type, "independence"),
        "interpretation": interp,
        "note": (
            "DW 假設殘差依資料列順序排列，且資料具時間或空間依賴結構。"
            "若資料列順序無實際意義（如隨機抽樣），DW 值的解讀需謹慎。"
        ),
    }


# ── 3. 等變異性假設 ────────────────────────────────────────────────────────────
def check_homoscedasticity(
    y_fitted: np.ndarray,
    residuals: np.ndarray,
    X_design: np.ndarray,
    model_type: str = "linear",
) -> dict:
    """Breusch–Pagan 檢定 + 殘差圖資料。"""
    bp_stat = bp_pvalue = None
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_design)
        bp_stat = float(bp_stat)
        bp_pvalue = float(bp_pvalue)
        if bp_pvalue < 0.05:
            interp = f"BP={bp_stat:.4f}, p={bp_pvalue:.4f}：拒絕等變異性假設，殘差可能存在異質變異（Heteroscedasticity）。"
        else:
            interp = f"BP={bp_stat:.4f}, p={bp_pvalue:.4f}：無足夠證據拒絕等變異性假設。"
    except Exception as e:
        interp = f"無法執行 Breusch–Pagan 檢定：{e}"

    return {
        "name": "等變異性（Homoscedasticity）",
        "method": "Residuals vs. Fitted Plot + Breusch–Pagan Test",
        "bp_statistic": bp_stat,
        "bp_pvalue": bp_pvalue,
        "y_fitted": y_fitted,
        "residuals": residuals,
        "applicability": _label(model_type, "homoscedasticity"),
        "interpretation": interp,
        "note": (
            "Breusch–Pagan 檢定基於 OLS 假設，用於 Ridge/Lasso 時為補充診斷。"
            "大樣本下即使輕微異質變異也可能顯著，需結合殘差圖判斷。"
        ),
    }


# ── 4. 殘差常態性假設 ──────────────────────────────────────────────────────────
def check_normality(
    residuals: np.ndarray,
    model_type: str = "linear",
) -> dict:
    """Q-Q Plot 資料 + Shapiro–Wilk（樣本 ≤ 5000）或 Anderson–Darling（大樣本）。"""
    from scipy import stats

    # Shapiro–Wilk 樣本限制
    sw_stat = sw_pvalue = None
    ad_stat = ad_critical = None
    large_sample_warning = False
    n = len(residuals)

    if n <= 5000:
        sw_stat, sw_pvalue = stats.shapiro(residuals[:5000])
        sw_stat, sw_pvalue = float(sw_stat), float(sw_pvalue)
        if sw_pvalue < 0.05:
            interp = f"Shapiro–Wilk p={sw_pvalue:.4f}：拒絕常態性假設（p < 0.05）。"
        else:
            interp = f"Shapiro–Wilk p={sw_pvalue:.4f}：無足夠證據拒絕常態性假設。"
    else:
        large_sample_warning = True
        # 對大樣本隨機取 5000 筆
        sample = residuals[np.random.choice(n, 5000, replace=False)]
        sw_stat, sw_pvalue = stats.shapiro(sample)
        sw_stat, sw_pvalue = float(sw_stat), float(sw_pvalue)
        interp = (
            f"⚠️ 樣本數 {n:,} 過大，Shapiro–Wilk 在大樣本下幾乎必然顯著（即使只有輕微偏態）。"
            f"以隨機 5000 筆子樣本計算：p={sw_pvalue:.4f}。"
            "建議優先參考 Q-Q Plot 判斷殘差分布型態。"
        )

    # Q-Q plot 資料
    theoretical_q, sample_q = stats.probplot(residuals, dist="norm", fit=False)

    return {
        "name": "殘差常態性（Normality）",
        "method": "Q-Q Plot + Shapiro–Wilk",
        "sw_statistic": sw_stat,
        "sw_pvalue": sw_pvalue,
        "large_sample_warning": large_sample_warning,
        "theoretical_quantiles": theoretical_q,
        "sample_quantiles": sample_q,
        "applicability": _label(model_type, "normality"),
        "interpretation": interp,
        "note": (
            "常態性假設主要影響傳統係數信賴區間與預測區間的精確度，"
            "不是預測準確度的必要條件。大樣本下（中央極限定理）可放寬此假設。"
        ),
    }


# ── 5. 多重共線性 ──────────────────────────────────────────────────────────────
def check_multicollinearity(
    X: pd.DataFrame,
    feature_names: list[str],
    model_type: str = "linear",
) -> dict:
    """VIF 計算 + 相關性矩陣。"""
    vif_df = None
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_arr = X[feature_names].dropna().values
        vif_values = []
        for i in range(X_arr.shape[1]):
            try:
                vif = variance_inflation_factor(X_arr, i)
            except Exception:
                vif = float("nan")
            vif_values.append(vif)

        vif_df = pd.DataFrame({
            "特徵": feature_names,
            "VIF": [round(v, 4) for v in vif_values],
        })
        vif_df["判斷"] = vif_df["VIF"].apply(
            lambda v: "⚠️ 高共線性（VIF > 10）" if v > 10
            else ("⚠️ 中等共線性（VIF 5-10）" if v > 5 else "✅ 正常")
        )
        high_vif = vif_df[vif_df["VIF"] > 10]["特徵"].tolist()
        if high_vif:
            interp = f"以下特徵 VIF > 10，存在高度多重共線性：{high_vif}"
        else:
            interp = "所有特徵 VIF ≤ 10，多重共線性無明顯問題。"
    except Exception as e:
        interp = f"VIF 計算失敗：{e}"

    corr_matrix = X[feature_names].corr()

    return {
        "name": "多重共線性（Multicollinearity）",
        "method": "VIF + Correlation Matrix",
        "vif_table": vif_df,
        "correlation_matrix": corr_matrix,
        "applicability": _label(model_type, "multicollinearity"),
        "interpretation": interp,
        "note": (
            "VIF 對 OLS 係數解釋尤其重要。Ridge/Lasso 透過正則化減緩共線性影響，"
            "但不會消除特徵間的相關性。"
        ),
    }


# ── 彙整入口 ──────────────────────────────────────────────────────────────────
def run_all_diagnostics(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    model_type: str = "linear",
) -> dict:
    """執行全部五項診斷，回傳 dict。"""
    y_pred_test = pipeline.predict(X_test)
    residuals = np.array(y_test) - y_pred_test
    y_fitted = y_pred_test

    # X_design for Breusch-Pagan
    preprocessor_steps = ["imputer", "scaler"]
    try:
        # 嘗試使用 pipeline 前段處理
        from sklearn.pipeline import Pipeline as SKPipeline
        prep = SKPipeline([
            (name, pipeline.named_steps[name])
            for name in preprocessor_steps if name in pipeline.named_steps
        ])
        X_design = prep.transform(X_test[feature_names])
    except Exception:
        X_design = X_test[feature_names].fillna(X_test[feature_names].median()).values

    return {
        "linearity":         check_linearity(y_fitted, residuals, model_type),
        "independence":      check_independence(residuals, model_type),
        "homoscedasticity":  check_homoscedasticity(y_fitted, residuals, X_design, model_type),
        "normality":         check_normality(residuals, model_type),
        "multicollinearity": check_multicollinearity(X_test, feature_names, model_type),
    }
