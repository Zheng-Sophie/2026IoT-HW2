"""
evaluation.py — 五項評估指標、模型比較表、最佳模型推薦
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import adjusted_r2, MODEL_NAMES


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_features: int,
) -> dict:
    """
    計算五項評估指標：MAE, MSE, RMSE, R², Adjusted R²。
    n_features: 模型實際使用的預測變數數量 p
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    p = n_features

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = adjusted_r2(r2, n, p)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adj_r2,
        "n_samples": n,
        "n_features": p,
    }


def evaluate_on_test(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_features: int,
) -> dict:
    """使用最終(已 fit)的 Pipeline 在測試集上評估。"""
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, n_features)
    metrics["y_pred"] = y_pred
    metrics["y_true"] = np.array(y_test)
    metrics["residuals"] = np.array(y_test) - y_pred
    return metrics


def build_comparison_table(
    results: list[dict],
) -> pd.DataFrame:
    """
    results: list of dict, 每個 dict 包含
             model_label、metrics (from compute_metrics)、source ('CV' 或 'Test')
    回傳方便展示的 DataFrame。
    """
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "模型": r["model_label"],
            "資料來源": r.get("source", "-"),
            "MAE": m["MAE"],
            "MSE": m["MSE"],
            "RMSE": m["RMSE"],
            "R²": m["R²"],
            "Adjusted R²": m["Adjusted R²"] if isinstance(m["Adjusted R²"], float) else m["Adjusted R²"],
            "特徵數": m["n_features"],
        })
    return pd.DataFrame(rows)


def recommend_best_model(
    cv_results: list[dict],
) -> tuple[dict, str]:
    """
    從 CV 結果中以透明規則推薦最佳模型。
    規則（優先順序）：
      1. CV RMSE 最低（主要準則）
      2. 相近 RMSE 時（差距 < 1%），優先選 R² 較高的
      3. 再相近時，優先選特徵數較少（解釋性較好）

    cv_results: list of dict，需包含 model_label, cv_rmse, cv_r2, n_features
    回傳 (best_result, reasoning_text)
    """
    if not cv_results:
        return None, "尚未有實驗結果。"

    sorted_by_rmse = sorted(cv_results, key=lambda x: x["cv_rmse"])
    best = sorted_by_rmse[0]

    reasoning_parts = [
        f"**{best['model_label']}** 在訓練集 Cross-Validation 中 RMSE 最低（{best['cv_rmse']:.6f}）。",
    ]

    # 檢查是否有 RMSE 非常接近的次優模型
    if len(sorted_by_rmse) > 1:
        second = sorted_by_rmse[1]
        rel_diff = abs(best["cv_rmse"] - second["cv_rmse"]) / (abs(best["cv_rmse"]) + 1e-12)
        if rel_diff < 0.01:
            reasoning_parts.append(
                f"次優模型 {second['model_label']} 的 RMSE 差距僅 {rel_diff*100:.2f}%，"
                "差距極小，可視解釋性需求選擇特徵數較少的模型。"
            )

    reasoning_parts += [
        f"CV R²：{best['cv_r2']:.6f}，使用特徵數：{best['n_features']}。",
        "",
        "⚠️ 注意：此推薦基於訓練集 CV，測試集結果請見最終評估頁面。",
        "CV 分數用於模型選擇參考，最終泛化效能請以測試集結果為準。",
    ]

    return best, "\n".join(reasoning_parts)
