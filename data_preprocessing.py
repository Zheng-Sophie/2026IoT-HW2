"""
data_preprocessing.py — 抽樣、切分與前處理（支援數值 + 類別混合型資料）

核心設計：
- 使用 ColumnTransformer 分別處理數值欄位與類別欄位
- 數值：SimpleImputer(median) + StandardScaler
- 類別：SimpleImputer(most_frequent) + OneHotEncoder
- 所有前處理只在訓練折 fit，避免資料洩漏
- 回傳 encoded 後的 DataFrame，欄位名稱可追溯
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DEFAULT_RANDOM_SEED, DEFAULT_SAMPLE_SIZE, DEFAULT_TRAIN_RATIO


def sample_data(
    df: pd.DataFrame,
    sample_size: int | str,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """依 sample_size 抽樣。若資料筆數 ≤ sample_size，使用全部資料。"""
    if sample_size == "全部資料" or sample_size >= len(df):
        return df.copy()
    return df.sample(n=int(sample_size), random_state=random_seed).reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """依 train_ratio 切分 X_train, X_test, y_train, y_test（原始欄位，未編碼）。"""
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=random_seed
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def build_mixed_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    建立支援混合型資料的 ColumnTransformer：
    - 數值欄位：中位數填補 + StandardScaler
    - 類別欄位：眾數填補 + OneHotEncoder（handle_unknown='ignore'）
    """
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """依 dtype 自動分類數值欄位與類別欄位。"""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def get_encoded_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """從已 fit 的 ColumnTransformer 取得編碼後的欄位名稱。"""
    names = list(numeric_cols)  # 數值欄位名稱不變
    if categorical_cols and "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        names += ohe_names
    return names


def preprocess_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, list[str], list[str]]:
    """
    在 X_train 上 fit ColumnTransformer，transform X_train 與 X_test。
    回傳 (X_train_enc_df, X_test_enc_df, fitted_preprocessor, numeric_cols, categorical_cols)
    回傳的 X 為 DataFrame，欄位名稱為編碼後的名稱。
    """
    numeric_cols, categorical_cols = get_column_types(X_train)
    preprocessor = build_mixed_preprocessor(numeric_cols, categorical_cols)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    encoded_names = get_encoded_feature_names(preprocessor, numeric_cols, categorical_cols)
    X_train_enc = pd.DataFrame(X_train_arr, columns=encoded_names, dtype=float)
    X_test_enc = pd.DataFrame(X_test_arr, columns=encoded_names, dtype=float)

    return X_train_enc, X_test_enc, preprocessor, numeric_cols, categorical_cols


def get_processed_data_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    orig_numeric_cols: list[str] | None = None,
    orig_categorical_cols: list[str] | None = None,
) -> dict:
    """回傳訓練集與測試集的統計摘要。"""
    return {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "orig_numeric_cols": orig_numeric_cols or [],
        "orig_categorical_cols": orig_categorical_cols or [],
        "train_target_mean": float(y_train.mean()),
        "train_target_std": float(y_train.std()),
        "test_target_mean": float(y_test.mean()),
        "test_target_std": float(y_test.std()),
    }
