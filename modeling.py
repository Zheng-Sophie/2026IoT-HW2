"""
modeling.py — 三種回歸模型 Pipeline（Linear, Ridge, Lasso）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso, LassoCV, LinearRegression, Ridge, RidgeCV,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CV_FOLDS, DEFAULT_RANDOM_SEED,
    LASSO_ALPHA_GRID, RIDGE_ALPHA_GRID,
)


def _base_steps():
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


class ColumnSelector:
    """選取指定欄位的 transformer（按欄位名稱）。"""
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def fit(self, X, y=None): return self

    def transform(self, X):
        if hasattr(X, "to_numpy"):
            # pandas DataFrame — 按名稱選取
            return X[self.feature_names].values
        return X  # 若已為 numpy array，直接回傳（假設已對齊）

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {"feature_names": self.feature_names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_pipeline_linear(feature_names: list[str]) -> Pipeline:
    return Pipeline(_base_steps() + [
        ("selector", ColumnSelector(feature_names)),
        ("model", LinearRegression()),
    ])


def build_pipeline_ridge(
    feature_names: list[str],
    alpha: float | None = None,
    cv: int = CV_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    if alpha is not None:
        model = Ridge(alpha=alpha)
    else:
        model = RidgeCV(alphas=RIDGE_ALPHA_GRID, cv=KFold(cv, shuffle=True, random_state=seed))
    return Pipeline(_base_steps() + [
        ("selector", ColumnSelector(feature_names)),
        ("model", model),
    ])


def build_pipeline_lasso(
    feature_names: list[str],
    alpha: float | None = None,
    cv: int = CV_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    if alpha is not None:
        model = Lasso(alpha=alpha, max_iter=10000, random_state=seed)
    else:
        model = LassoCV(
            alphas=LASSO_ALPHA_GRID, cv=KFold(cv, shuffle=True, random_state=seed),
            max_iter=10000, random_state=seed,
        )
    return Pipeline(_base_steps() + [
        ("selector", ColumnSelector(feature_names)),
        ("model", model),
    ])


def fit_model(
    model_type: str,
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[Pipeline, dict]:
    """
    訓練指定模型，回傳 (fitted_pipeline, info_dict)。
    info_dict 包含係數、截距、選中 alpha 等資訊。
    """
    if model_type == "linear":
        pipeline = build_pipeline_linear(feature_names)
    elif model_type == "ridge":
        pipeline = build_pipeline_ridge(feature_names, seed=seed)
    elif model_type == "lasso":
        pipeline = build_pipeline_lasso(feature_names, seed=seed)
    else:
        raise ValueError(f"未知模型類型：{model_type}")

    pipeline.fit(X_train, y_train)
    model_obj = pipeline.named_steps["model"]

    info = {
        "model_type": model_type,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }

    # 提取係數
    if hasattr(model_obj, "coef_"):
        info["coefficients"] = dict(zip(feature_names, model_obj.coef_.tolist()))
        info["intercept"] = float(model_obj.intercept_) if hasattr(model_obj, "intercept_") else None

    # 提取 Ridge alpha
    if model_type == "ridge":
        if hasattr(model_obj, "alpha_"):
            info["best_alpha"] = float(model_obj.alpha_)
        elif hasattr(model_obj, "alpha"):
            info["best_alpha"] = float(model_obj.alpha)

    # 提取 Lasso alpha + non-zero coefs
    if model_type == "lasso":
        if hasattr(model_obj, "alpha_"):
            info["best_alpha"] = float(model_obj.alpha_)
        elif hasattr(model_obj, "alpha"):
            info["best_alpha"] = float(model_obj.alpha)
        non_zero = sum(1 for v in model_obj.coef_ if v != 0)
        info["non_zero_features"] = non_zero

    return pipeline, info


def cv_evaluate_model(
    model_type: str,
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = CV_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """
    對訓練集做 k-fold CV，回傳平均 RMSE 與 R²。
    使用不含 alpha 搜尋的固定 pipeline（避免 nested CV 複雜度）。
    注意：此 CV 結果用於特徵數選擇；最終測試集評估在 evaluation.py 完成。
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    if model_type == "linear":
        pipeline = build_pipeline_linear(feature_names)
    elif model_type == "ridge":
        # 使用固定 alpha=1.0 作為快速 CV 估計
        pipeline = build_pipeline_ridge(feature_names, alpha=1.0, seed=seed)
    elif model_type == "lasso":
        pipeline = build_pipeline_lasso(feature_names, alpha=0.01, seed=seed)
    else:
        raise ValueError(f"未知模型類型：{model_type}")

    neg_mse = cross_val_score(pipeline, X_train, y_train,
                              cv=kf, scoring="neg_mean_squared_error")
    r2_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=kf, scoring="r2")
    rmse = np.sqrt(-neg_mse)
    return {
        "cv_rmse_mean": float(rmse.mean()),
        "cv_rmse_std": float(rmse.std()),
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
    }
