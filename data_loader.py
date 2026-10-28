"""
data_loader.py — 資料集載入（預設 Predict the Flood + 自訂 URL + 本機上傳）
"""
from __future__ import annotations

import io
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import (
    ALLOWED_URL_SCHEMES,
    DEFAULT_DATASET_LOCAL_FILENAME,
    DEFAULT_DATASET_NAME,
    DEFAULT_TARGET_COLUMN,
    ID_COLUMN_PATTERNS,
    KAGGLE_COMPETITION_URL,
    MAX_REDIRECTS,
    URL_DOWNLOAD_TIMEOUT_SEC,
    URL_MAX_BYTES,
)


# ── 清除舊結果 ────────────────────────────────────────────────────────────────
CLEARABLE_KEYS = [
    "processed_data",
    "feature_selection_results",
    "experiment_results",
    "trained_models",
    "evaluation_results",
    "diagnostics_results",
    "best_model_key",
    "prediction_pipeline",
]


def clear_derived_results() -> None:
    """切換資料集後，清除所有依賴舊資料集的計算結果。"""
    for key in CLEARABLE_KEYS:
        if key in st.session_state:
            del st.session_state[key]


# ── 預設資料集 ────────────────────────────────────────────────────────────────
def load_default_dataset() -> tuple[pd.DataFrame | None, str]:
    """
    嘗試載入 Predict the Flood 資料集。
    優先使用與 app.py 同目錄的 train.csv。
    若找不到，回傳 None 並附說明。
    """
    local_path = Path(__file__).parent / DEFAULT_DATASET_LOCAL_FILENAME
    if local_path.exists():
        try:
            df = pd.read_csv(local_path)
            return df, f"✅ 已從本機載入：{local_path.name}"
        except Exception as e:
            return None, f"❌ 讀取本機檔案失敗：{e}"
    return None, "📁 尚未找到本機資料集"


def get_dataset_load_instructions() -> str:
    return f"""
**如何取得 Student Performance Factors 資料集：**

1. 前往：{KAGGLE_COMPETITION_URL}
2. 登入 Kaggle 帳號後下載 `StudentPerformanceFactors.csv`
3. 將 `StudentPerformanceFactors.csv` 放置於本應用程式同一目錄（`{Path(__file__).parent}`）
4. 重新整理本頁面

或者，您可以在下方使用「上傳本機 CSV」功能直接上傳。
"""


# ── 自訂 URL 載入 ─────────────────────────────────────────────────────────────
def _validate_url(url: str) -> tuple[bool, str]:
    """基本安全檢查，不執行實際下載。"""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "無法解析網址格式"

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        return False, f"不支援的協定：{parsed.scheme}（僅支援 http / https）"

    hostname = parsed.hostname or ""
    # 防止存取內網 / 本機
    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if hostname in blocked_hosts:
        return False, "不允許存取本機或內部網路位址"
    private_prefixes = ("10.", "192.168.", "172.")
    if any(hostname.startswith(p) for p in private_prefixes):
        return False, "不允許存取私有網路位址"

    return True, ""


def load_from_url(url: str) -> tuple[pd.DataFrame | None, str]:
    """從公開 CSV URL 下載並解析資料。"""
    ok, reason = _validate_url(url)
    if not ok:
        return None, f"❌ 網址驗證失敗：{reason}"

    try:
        response = requests.get(
            url,
            timeout=URL_DOWNLOAD_TIMEOUT_SEC,
            allow_redirects=True,
            stream=True,
            headers={"User-Agent": "RegressionLab/1.0"},
        )
    except requests.exceptions.SSLError:
        return None, "❌ SSL 憑證驗證失敗"
    except requests.exceptions.ConnectionError:
        return None, "❌ 無法連線：請確認網址是否正確或網路是否正常"
    except requests.exceptions.Timeout:
        return None, f"❌ 連線逾時（{URL_DOWNLOAD_TIMEOUT_SEC} 秒）"
    except Exception as e:
        return None, f"❌ 連線失敗：{e}"

    if response.status_code == 401 or response.status_code == 403:
        return None, f"❌ 需要認證（HTTP {response.status_code}）：Kaggle 等平台需要登入或 API 憑證"
    if response.status_code == 404:
        return None, "❌ 找不到資源（HTTP 404）"
    if response.status_code != 200:
        return None, f"❌ HTTP 錯誤：{response.status_code}"

    content_type = response.headers.get("Content-Type", "")
    if "html" in content_type.lower():
        return None, "❌ 網址指向 HTML 網頁，不是 CSV 檔案。GitHub 請使用 raw file URL（raw.githubusercontent.com）"

    # 讀取內容（限制大小）
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > URL_MAX_BYTES:
            return None, f"❌ 檔案過大（超過 {URL_MAX_BYTES // 1024 // 1024} MB 限制）"
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return None, f"❌ CSV 解析失敗：{e}"

    if df.empty:
        return None, "❌ 載入的 CSV 是空的"

    return df, f"✅ 已從 URL 載入（{len(df):,} 列）"


# ── 上傳本機 CSV ──────────────────────────────────────────────────────────────
def load_from_uploaded_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """從 Streamlit UploadedFile 解析 CSV。"""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"❌ CSV 解析失敗：{e}"

    if df.empty:
        return None, "❌ 上傳的 CSV 是空的"

    return df, f"✅ 已載入上傳檔案：{uploaded_file.name}（{len(df):,} 列）"


# ── 資料集指紋 ────────────────────────────────────────────────────────────────
def dataset_fingerprint(df: pd.DataFrame) -> str:
    """產生可作為快取 key 的資料集指紋（shape + 前幾個值的 hash）。"""
    import hashlib
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    sample_str = str(df.iloc[:5].values.tolist())
    col_str = ",".join(df.columns.tolist())
    raw = f"{shape_str}|{col_str}|{sample_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
