"""
prediction_intervals.py — 95% 預測區間

OLS:  使用 statsmodels OLS get_prediction() 計算傳統預測區間。
Ridge/Lasso: 標示為「尚未支援」，不套用 OLS 公式。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_ols_prediction_interval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_new: pd.DataFrame,
    feature_names: list[str],
    alpha: float = 0.05,
) -> dict:
    """
    使用 statsmodels OLS 計算 95% 預測區間（個別觀測，非均值）。

    注意：
    - 此公式基於誤差常態性、等變異性假設。
    - Prediction Interval 比 Confidence Interval 寬，包含個別觀測誤差。
    - 若 X_new 遠超訓練資料範圍，區間精確度降低。
    """
    try:
        import statsmodels.api as sm

        X_tr = X_train[feature_names].copy()
        X_new_sel = X_new[feature_names].copy()

        # 填補缺失值（與訓練一致）
        col_medians = X_tr.median()
        X_tr = X_tr.fillna(col_medians)
        X_new_sel = X_new_sel.fillna(col_medians)

        # 加入截距
        X_tr_const = sm.add_constant(X_tr.values, has_constant="add")
        X_new_const = sm.add_constant(X_new_sel.values, has_constant="add")

        model = sm.OLS(y_train.values, X_tr_const)
        result = model.fit()

        pred = result.get_prediction(X_new_const)
        pi = pred.summary_frame(alpha=alpha)

        return {
            "supported": True,
            "method": "statsmodels OLS Prediction Interval",
            "y_pred": pi["mean"].values,
            "pi_lower": pi["obs_ci_lower"].values,
            "pi_upper": pi["obs_ci_upper"].values,
            "ci_lower": pi["mean_ci_lower"].values,
            "ci_upper": pi["mean_ci_upper"].values,
            "alpha": alpha,
            "note": (
                "95% 預測區間（個別觀測）：包含模型不確定性及個別觀測誤差，"
                "通常比信賴區間（均值）寬。"
                "此區間基於誤差常態性與等變異性假設。"
            ),
        }
    except Exception as e:
        return {
            "supported": False,
            "method": "statsmodels OLS Prediction Interval",
            "error": str(e),
            "note": f"OLS 預測區間計算失敗：{e}",
        }


def get_ridge_lasso_pi_status(model_type: str) -> dict:
    """
    Ridge / Lasso 不支援直接套用 OLS 預測區間公式。
    回傳狀態說明，不計算任何區間。
    """
    return {
        "supported": False,
        "method": f"{model_type.capitalize()} Prediction Interval",
        "note": (
            f"⚠️ {model_type.capitalize()} Regression 不支援直接套用普通 OLS 的傳統預測區間公式。"
            "正則化改變了係數估計的分布，標準 OLS 區間在此不具正確涵蓋率保證。"
            "如需 Ridge/Lasso 的不確定性量化，可考慮 Bootstrap 方法，但第一版尚未實作。"
        ),
    }


def compute_prediction_interval(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_new: pd.DataFrame,
    feature_names: list[str],
    model_type: str = "linear",
    alpha: float = 0.05,
) -> dict:
    """統一入口：依模型類型決定使用 OLS PI 或標示不支援。"""
    if model_type == "linear":
        return get_ols_prediction_interval(
            X_train, y_train, X_new, feature_names, alpha
        )
    else:
        return get_ridge_lasso_pi_status(model_type)


def check_out_of_range(
    X_new: pd.DataFrame,
    X_train: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 3.0,
) -> dict[str, str]:
    """
    檢查 X_new 的特徵值是否超出訓練集的常見範圍（以 z-score 判斷）。
    回傳 {feature: warning_message} for out-of-range features。
    """
    warnings = {}
    for col in feature_names:
        if col not in X_train.columns or col not in X_new.columns:
            continue
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()
        if train_std < 1e-10:
            continue
        val = float(X_new[col].iloc[0])
        z = abs(val - train_mean) / train_std
        if z > threshold:
            warnings[col] = (
                f"輸入值 {val:.4f} 超出訓練資料常見範圍 "
                f"（訓練集：{train_mean:.4f} ± {train_std:.4f}，z={z:.1f}）"
            )
    return warnings
