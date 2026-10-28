"""
data_validation.py — 欄位、缺失值、型態及資料品質檢查
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from config import ID_COLUMN_PATTERNS


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    對載入的 DataFrame 執行全面品質檢查。
    回傳結構化報告 dict。
    """
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": {},
        "duplicate_rows": 0,
        "constant_cols": [],
        "likely_id_cols": [],
        "numeric_cols": [],
        "non_numeric_cols": [],
        "warnings": [],
        "errors": [],
    }

    # 缺失值
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing[col] > 0
    }

    # 重複列
    report["duplicate_rows"] = int(df.duplicated().sum())
    if report["duplicate_rows"] > 0:
        report["warnings"].append(
            f"發現 {report['duplicate_rows']:,} 列重複紀錄"
        )

    # 欄位型態分類
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    report["numeric_cols"] = numeric_cols
    report["non_numeric_cols"] = non_numeric_cols
    if non_numeric_cols:
        report["warnings"].append(
            f"非數値欄位（將以 One-Hot Encoding 處理）：{non_numeric_cols}"
        )

    # 常數欄位
    constant_cols = [
        c for c in numeric_cols if df[c].nunique(dropna=True) <= 1
    ]
    report["constant_cols"] = constant_cols
    if constant_cols:
        report["warnings"].append(f"常數欄位（無變異量）：{constant_cols}")

    # 可能的 ID 欄位
    likely_id = []
    for col in df.columns:
        col_lower = col.lower()
        if any(col_lower == p.lower() for p in ID_COLUMN_PATTERNS):
            likely_id.append(col)
        elif col_lower.endswith("_id") or col_lower.startswith("id_"):
            likely_id.append(col)
        elif df[col].dtype in [np.int64, np.int32, np.float64]:
            # 若值唯一且連續遞增，視為 ID
            if df[col].nunique() == len(df) and df[col].is_monotonic_increasing:
                likely_id.append(col)
    report["likely_id_cols"] = list(set(likely_id))
    if report["likely_id_cols"]:
        report["warnings"].append(
            f"可能的 ID 欄位（建議排除）：{report['likely_id_cols']}"
        )

    # 高缺失欄位
    high_missing = [
        col for col, v in report["missing"].items() if v["pct"] > 50
    ]
    if high_missing:
        report["warnings"].append(
            f"缺失值超過 50% 的欄位：{high_missing}"
        )

    return report


def validate_target_column(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
    """
    確認目標欄位適合回歸：
    - 欄位存在
    - 為數值型態
    - 有足夠的唯一值（> 2）
    回傳 (is_valid, reason)
    """
    if target_col not in df.columns:
        return False, f"欄位 `{target_col}` 不存在於資料集中"

    col_dtype = df[target_col].dtype
    if not pd.api.types.is_numeric_dtype(col_dtype):
        return False, (
            f"欄位 `{target_col}` 的型態為 {col_dtype}，不適合回歸。"
            f"請選擇數值型欄位。"
        )

    n_unique = df[target_col].nunique(dropna=True)
    if n_unique <= 2:
        return False, (
            f"欄位 `{target_col}` 只有 {n_unique} 個唯一值，"
            "疑似為分類標籤，不適合多元線性回歸。"
        )

    n_missing = df[target_col].isnull().sum()
    if n_missing / len(df) > 0.5:
        return False, (
            f"目標欄位 `{target_col}` 缺失值超過 50%（{n_missing:,} 列），"
            "資料不足以進行有效回歸。"
        )

    return True, "✅ 目標欄位驗證通過"


def get_candidate_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """
    從 DataFrame 中找出適合作為特徵的欄位：
    - 排除目標欄位
    - 排除使用者指定排除欄位
    - 保留數値欄位 + 類別欄位（字串/category）——後者將由 OneHotEncoder 處理
    """
    exclude = set([target_col] + (exclude_cols or []))
    usable = df.select_dtypes(include=[np.number, "object", "category"]).columns.tolist()
    return [c for c in usable if c not in exclude]


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """數值欄位描述統計（與 pandas describe 相同，格式化為中文欄位名）。"""
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats.columns = ["筆數", "平均值", "標準差", "最小值", "Q1", "中位數", "Q3", "最大值"]
    return stats.round(4)
