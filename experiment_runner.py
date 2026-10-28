"""
experiment_runner.py — 實驗框架：特徵數曲線、全實驗、快速實驗
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import (
    CV_FOLDS, DEFAULT_RANDOM_SEED, FIXED_FEATURE_COUNTS,
    FEATURE_SELECTION_METHODS, MODEL_NAMES,
)
from feature_selection import run_feature_selection, SelectionResult
from modeling import cv_evaluate_model


@dataclass
class ExperimentPoint:
    """單一（方法, 特徵數, 模型）實驗結果。"""
    method_key: str
    method_name: str
    n_features: int
    model_key: str
    model_name: str
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    selected_features: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0


@dataclass
class ExperimentResults:
    points: list[ExperimentPoint] = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        if not self.points:
            return pd.DataFrame()
        rows = [
            {
                "特徵選擇方法": p.method_name,
                "特徵數": p.n_features,
                "模型": p.model_name,
                "CV RMSE (mean)": round(p.cv_rmse_mean, 6),
                "CV RMSE (std)": round(p.cv_rmse_std, 6),
                "CV R² (mean)": round(p.cv_r2_mean, 6),
                "CV R² (std)": round(p.cv_r2_std, 6),
                "耗時(秒)": round(p.elapsed_sec, 2),
            }
            for p in self.points
        ]
        return pd.DataFrame(rows)

    def filter(self, method_key=None, model_key=None) -> "ExperimentResults":
        pts = self.points
        if method_key:
            pts = [p for p in pts if p.method_key == method_key]
        if model_key:
            pts = [p for p in pts if p.model_key == model_key]
        return ExperimentResults(points=pts)

    def get_n_features_curve(
        self, method_key: str, model_key: str
    ) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
        """回傳 (n_features_list, rmse_means, rmse_stds, r2_means, r2_stds)"""
        pts = sorted(
            [p for p in self.points if p.method_key == method_key and p.model_key == model_key],
            key=lambda p: p.n_features,
        )
        return (
            [p.n_features for p in pts],
            [p.cv_rmse_mean for p in pts],
            [p.cv_rmse_std for p in pts],
            [p.cv_r2_mean for p in pts],
            [p.cv_r2_std for p in pts],
        )


def _get_feature_counts(
    max_features: int, mode: str, custom_counts: list[int] | None = None
) -> list[int]:
    """
    mode='fixed': 使用 FIXED_FEATURE_COUNTS，超過 max 的截斷。
    mode='auto':  從 1 到 max_features 逐步（但可縮減步距避免過多點）。
    mode='custom': 使用 custom_counts。
    """
    if mode == "fixed":
        counts = [k for k in FIXED_FEATURE_COUNTS if k <= max_features]
        if not counts:
            counts = list(range(1, min(max_features + 1, 6)))
        return counts
    elif mode == "auto":
        if max_features <= 10:
            return list(range(1, max_features + 1))
        step = max(1, max_features // 10)
        counts = list(range(step, max_features + 1, step))
        if max_features not in counts:
            counts.append(max_features)
        return sorted(set(counts))
    elif mode == "custom" and custom_counts:
        return sorted([k for k in custom_counts if 1 <= k <= max_features])
    return [min(k, max_features) for k in FIXED_FEATURE_COUNTS]


def run_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method_keys: list[str],
    model_keys: list[str],
    feature_count_mode: str = "fixed",
    custom_counts: list[int] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> ExperimentResults:
    """
    執行指定方法 × 特徵數 × 模型的所有實驗組合。

    progress_callback(done, total, msg): 給 Streamlit st.progress 用。
    所有特徵選擇均在 CV 內執行，防止資料洩漏。
    """
    max_features = X_train.shape[1]
    feature_counts = _get_feature_counts(max_features, feature_count_mode, custom_counts)

    results = ExperimentResults()
    total = len(method_keys) * len(feature_counts) * len(model_keys)
    done = 0

    for method_key in method_keys:
        for n_k in feature_counts:
            t0 = time.time()
            try:
                sel_result: SelectionResult = run_feature_selection(
                    method_key, X_train, y_train, n_k, random_seed
                )
                selected = sel_result.selected_features
            except Exception as e:
                # 特徵選擇失敗，跳過這個特徵數
                done += len(model_keys)
                if progress_callback:
                    progress_callback(done, total, f"⚠️ {method_key} k={n_k} 失敗：{e}")
                continue

            for model_key in model_keys:
                t1 = time.time()
                try:
                    cv_info = cv_evaluate_model(
                        model_key, selected, X_train, y_train,
                        cv=CV_FOLDS, seed=random_seed,
                    )
                    point = ExperimentPoint(
                        method_key=method_key,
                        method_name=FEATURE_SELECTION_METHODS.get(method_key, method_key),
                        n_features=len(selected),
                        model_key=model_key,
                        model_name=MODEL_NAMES.get(model_key, model_key),
                        cv_rmse_mean=cv_info["cv_rmse_mean"],
                        cv_rmse_std=cv_info["cv_rmse_std"],
                        cv_r2_mean=cv_info["cv_r2_mean"],
                        cv_r2_std=cv_info["cv_r2_std"],
                        selected_features=selected,
                        elapsed_sec=time.time() - t1,
                    )
                    results.points.append(point)
                except Exception as e:
                    pass  # 靜默跳過，避免一個組合失敗中斷整批

                done += 1
                if progress_callback:
                    method_name = FEATURE_SELECTION_METHODS.get(method_key, method_key)
                    model_name = MODEL_NAMES.get(model_key, model_key)
                    progress_callback(
                        done, total,
                        f"{method_name} | k={n_k} | {model_name}"
                    )

    return results
