"""
feature_selection.py — 五種特徵選擇方法（均放在 CV 內以防洩漏）

每種方法回傳:
  SelectionResult 包含:
    method_name     : str
    selected_features: list[str]
    feature_scores  : dict[str, float] | None
    n_selected      : int
    cv_rmse         : float
    cv_r2           : float
    elapsed_sec     : float
    notes           : str
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SequentialFeatureSelector,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CV_FOLDS, DEFAULT_RANDOM_SEED, LASSO_ALPHA_GRID


@dataclass
class SelectionResult:
    method_name: str
    method_key: str
    selected_features: list[str]
    feature_scores: Optional[dict] = None   # {feature: score/importance}
    n_selected: int = 0
    cv_rmse: float = float("nan")
    cv_r2: float = float("nan")
    cv_rmse_std: float = float("nan")
    cv_r2_std: float = float("nan")
    elapsed_sec: float = 0.0
    notes: str = ""

    def __post_init__(self):
        self.n_selected = len(self.selected_features)


def _make_cv_pipeline(feature_mask_or_selector, model) -> Pipeline:
    """建立包含前處理、特徵篩選、模型的 Pipeline。"""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", feature_mask_or_selector),
        ("model", model),
    ])


def _cv_scores(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
               cv: int = CV_FOLDS, seed: int = DEFAULT_RANDOM_SEED
               ) -> tuple[float, float, float, float]:
    """回傳 (mean_rmse, std_rmse, mean_r2, std_r2)"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=kf,
                                     scoring="neg_mean_squared_error")
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    rmse_scores = np.sqrt(-neg_mse_scores)
    return (float(rmse_scores.mean()), float(rmse_scores.std()),
            float(r2_scores.mean()), float(r2_scores.std()))


# ── 1. SelectKBest ────────────────────────────────────────────────────────────
def select_kbest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SelectionResult:
    t0 = time.time()
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    pipeline = _make_cv_pipeline(selector, LinearRegression())
    mean_rmse, std_rmse, mean_r2, std_r2 = _cv_scores(pipeline, X_train, y_train)

    # fit once on all training data to get selected features
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_train)
    selector_fit = SelectKBest(score_func=f_regression, k=k)
    selector_fit.fit(X_imp, y_train)
    mask = selector_fit.get_support()
    selected = [col for col, m in zip(X_train.columns, mask) if m]
    scores = dict(zip(X_train.columns, selector_fit.scores_))

    return SelectionResult(
        method_name="SelectKBest",
        method_key="selectkbest",
        selected_features=selected,
        feature_scores=scores,
        cv_rmse=mean_rmse, cv_rmse_std=std_rmse,
        cv_r2=mean_r2, cv_r2_std=std_r2,
        elapsed_sec=time.time() - t0,
        notes=f"使用 f_regression 分數選取前 {k} 個特徵。",
    )


# ── 2. RFE ────────────────────────────────────────────────────────────────────
def select_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SelectionResult:
    t0 = time.time()
    k = min(k, X_train.shape[1])
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    pipeline = _make_cv_pipeline(rfe, LinearRegression())
    mean_rmse, std_rmse, mean_r2, std_r2 = _cv_scores(pipeline, X_train, y_train)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(imp.fit_transform(X_train))
    rfe_fit = RFE(estimator=LinearRegression(), n_features_to_select=k, step=1)
    rfe_fit.fit(X_proc, y_train)
    mask = rfe_fit.support_
    selected = [col for col, m in zip(X_train.columns, mask) if m]
    scores = {col: float(r) for col, r in zip(X_train.columns, rfe_fit.ranking_)}

    return SelectionResult(
        method_name="RFE",
        method_key="rfe",
        selected_features=selected,
        feature_scores=scores,
        cv_rmse=mean_rmse, cv_rmse_std=std_rmse,
        cv_r2=mean_r2, cv_r2_std=std_r2,
        elapsed_sec=time.time() - t0,
        notes=f"Ranking 越小越重要（1 為最佳）。選取前 {k} 個特徵。",
    )


# ── 3. Sequential Feature Selection (SFS) ────────────────────────────────────
def select_sfs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
    fast_mode: bool = True,
) -> SelectionResult:
    """
    fast_mode=True: 使用 LinearRegression 作為基礎模型，加速計算。
    """
    t0 = time.time()
    k = min(k, X_train.shape[1])
    base_estimator = LinearRegression()
    cv_inner = KFold(n_splits=3 if fast_mode else CV_FOLDS,
                     shuffle=True, random_state=random_seed)

    sfs = SequentialFeatureSelector(
        base_estimator,
        n_features_to_select=k,
        direction="forward",
        scoring="neg_mean_squared_error",
        cv=cv_inner,
        n_jobs=-1,
    )
    pipeline = _make_cv_pipeline(sfs, LinearRegression())
    mean_rmse, std_rmse, mean_r2, std_r2 = _cv_scores(pipeline, X_train, y_train)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(imp.fit_transform(X_train))
    sfs_fit = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=k,
        direction="forward",
        scoring="neg_mean_squared_error",
        cv=cv_inner,
        n_jobs=-1,
    )
    sfs_fit.fit(X_proc, y_train)
    mask = sfs_fit.get_support()
    selected = [col for col, m in zip(X_train.columns, mask) if m]

    mode_note = "快速模式（3-fold 內部 CV）" if fast_mode else "完整模式（5-fold 內部 CV）"
    return SelectionResult(
        method_name="Sequential Feature Selection",
        method_key="sfs",
        selected_features=selected,
        feature_scores=None,
        cv_rmse=mean_rmse, cv_rmse_std=std_rmse,
        cv_r2=mean_r2, cv_r2_std=std_r2,
        elapsed_sec=time.time() - t0,
        notes=f"逐步前向選擇，{mode_note}，選取 {k} 個特徵。",
    )


# ── 4. Lasso Feature Selection ────────────────────────────────────────────────
def select_lasso_fs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SelectionResult:
    """
    使用 LassoCV 找出在訓練集上非零係數的特徵，
    然後取係數絕對值最大的前 k 個。
    注意：此 Lasso 僅用於特徵選擇，與最終 Lasso 回歸模型不同。
    """
    t0 = time.time()
    k = min(k, X_train.shape[1])

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(imp.fit_transform(X_train))

    lasso_cv = LassoCV(
        alphas=LASSO_ALPHA_GRID,
        cv=CV_FOLDS,
        random_state=random_seed,
        max_iter=5000,
    )
    lasso_cv.fit(X_proc, y_train)
    coefs = np.abs(lasso_cv.coef_)
    coef_dict = {col: float(c) for col, c in zip(X_train.columns, coefs)}

    # 選取係數絕對值最大的前 k 個
    sorted_features = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, v in sorted_features[:k] if v > 0]
    # 若非零特徵不足 k 個，補足到 k 個（取係數最大的）
    if len(selected) < k:
        extra = [f for f, v in sorted_features if f not in selected]
        selected += extra[: k - len(selected)]
    selected = selected[:k]

    # 建立 Pipeline 做 CV 評估
    from sklearn.feature_selection import SelectFromModel
    lasso_est = Lasso(alpha=lasso_cv.alpha_, random_state=random_seed, max_iter=5000)

    class TopKLassoSelector:
        """自訂 transformer：選取係數絕對值最大的前 k 個特徵。"""
        def __init__(self, n: int, alpha: float, seed: int):
            self.n = n
            self.alpha = alpha
            self.seed = seed
            self.mask_ = None

        def fit(self, X, y):
            lasso = Lasso(alpha=self.alpha, random_state=self.seed, max_iter=5000)
            lasso.fit(X, y)
            coef_abs = np.abs(lasso.coef_)
            top_k_idx = np.argsort(coef_abs)[::-1][: self.n]
            self.mask_ = np.zeros(X.shape[1], dtype=bool)
            self.mask_[top_k_idx] = True
            return self

        def transform(self, X):
            return X[:, self.mask_]

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

        def get_params(self, deep=True):
            return {"n": self.n, "alpha": self.alpha, "seed": self.seed}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    selector_obj = TopKLassoSelector(k, lasso_cv.alpha_, random_seed)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", selector_obj),
        ("model", LinearRegression()),
    ])
    mean_rmse, std_rmse, mean_r2, std_r2 = _cv_scores(pipeline, X_train, y_train)

    non_zero = sum(1 for v in coef_dict.values() if v > 0)
    return SelectionResult(
        method_name="Lasso Feature Selection",
        method_key="lasso_fs",
        selected_features=selected,
        feature_scores=coef_dict,
        cv_rmse=mean_rmse, cv_rmse_std=std_rmse,
        cv_r2=mean_r2, cv_r2_std=std_r2,
        elapsed_sec=time.time() - t0,
        notes=(
            f"LassoCV 最佳 alpha={lasso_cv.alpha_:.4f}，"
            f"非零係數特徵：{non_zero} 個；"
            f"此 Lasso 僅用於特徵篩選，與最終 Lasso 回歸模型不同。"
        ),
    )


# ── 5. Permutation Importance ─────────────────────────────────────────────────
def select_permutation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SelectionResult:
    """
    在訓練集 validation fold 估計 Permutation Importance，
    選取重要性最高的前 k 個特徵。
    """
    t0 = time.time()
    k = min(k, X_train.shape[1])

    # 用一折 val set 估計 importance（不用全部訓練資料）
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_seed)
    importances_list = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_p = scaler.fit_transform(imp.fit_transform(X_tr))
        X_val_p = scaler.transform(imp.transform(X_val))

        model = LinearRegression()
        model.fit(X_tr_p, y_tr)

        pi = permutation_importance(
            model, X_val_p, y_val,
            n_repeats=5, random_state=random_seed,
            scoring="neg_mean_squared_error",
        )
        importances_list.append(pi.importances_mean)

    mean_importance = np.mean(importances_list, axis=0)
    coef_dict = {col: float(v) for col, v in zip(X_train.columns, mean_importance)}
    sorted_features = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in sorted_features[:k]]

    # CV score with selected features
    class FixedColumnSelector:
        def __init__(self, cols, feature_names):
            self.cols = list(cols)
            self.feature_names = list(feature_names)

        def fit(self, X, y=None): return self

        def transform(self, X):
            indices = [list(self.feature_names).index(c) for c in self.cols]
            return X[:, indices]

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

        def get_params(self, deep=True): return {}
        def set_params(self, **p): return self

    selector_obj = FixedColumnSelector(selected, X_train.columns.tolist())
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", selector_obj),
        ("model", LinearRegression()),
    ])
    mean_rmse, std_rmse, mean_r2, std_r2 = _cv_scores(pipeline, X_train, y_train)

    return SelectionResult(
        method_name="Permutation Importance",
        method_key="permutation",
        selected_features=selected,
        feature_scores=coef_dict,
        cv_rmse=mean_rmse, cv_rmse_std=std_rmse,
        cv_r2=mean_r2, cv_r2_std=std_r2,
        elapsed_sec=time.time() - t0,
        notes=(
            f"在 {CV_FOLDS}-fold 的 validation fold 上估計 Permutation Importance，"
            f"取前 {k} 個特徵。Importance 值為負 MSE 降低量（越大越重要）。"
        ),
    )


# ── 分派入口 ──────────────────────────────────────────────────────────────────
SELECTOR_MAP = {
    "selectkbest": select_kbest,
    "rfe": select_rfe,
    "sfs": select_sfs,
    "lasso_fs": select_lasso_fs,
    "permutation": select_permutation,
}


def run_feature_selection(
    method_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
    **kwargs,
) -> SelectionResult:
    if method_key not in SELECTOR_MAP:
        raise ValueError(f"未知的特徵選擇方法：{method_key}")
    return SELECTOR_MAP[method_key](X_train, y_train, k, random_seed, **kwargs)
