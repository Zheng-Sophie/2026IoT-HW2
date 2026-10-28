"""
config.py — RegressionLab 全域設定與常數
"""

# ── 預設資料集 ────────────────────────────────────────────────────
DEFAULT_DATASET_NAME = "Student Performance Factors"
DEFAULT_DATASET_LOCAL_FILENAME = "StudentPerformanceFactors.csv"
DEFAULT_TARGET_COLUMN = "Exam_Score"

# Kaggle 資料集參考頁面 (不可直接下載，須登入)
KAGGLE_COMPETITION_URL = "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"

# 不列入候選特徵的欄位（常見 ID 欄位模式）
ID_COLUMN_PATTERNS = ["id", "ID", "Id", "index", "Index"]

# ── 資料準備 ──────────────────────────────────────────────────────
DEFAULT_SAMPLE_SIZE_OPTIONS = [5_000, 20_000, 50_000, "全部資料"]
DEFAULT_SAMPLE_SIZE = 50_000          # 預設抽樣量
DEFAULT_RANDOM_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8            # 80/20 切分

# ── 特徵選擇 ──────────────────────────────────────────────────────
FIXED_FEATURE_COUNTS = [14, 16, 18, 20]   # 固定特徵數候選值
MIN_FEATURE_COUNT = 1

# ── 模型與訓練 ────────────────────────────────────────────────────
CV_FOLDS = 5
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LASSO_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

MODEL_NAMES = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
}

FEATURE_SELECTION_METHODS = {
    "selectkbest": "SelectKBest",
    "rfe": "RFE（Recursive Feature Elimination）",
    "sfs": "Sequential Feature Selection（SFS）",
    "lasso_fs": "Lasso Feature Selection",
    "permutation": "Permutation Importance",
}

# ── URL 安全限制 ──────────────────────────────────────────────────
URL_DOWNLOAD_TIMEOUT_SEC = 30
URL_MAX_BYTES = 200 * 1024 * 1024    # 200 MB
ALLOWED_URL_SCHEMES = ["http", "https"]
MAX_REDIRECTS = 5

# ── Adjusted R² ──────────────────────────────────────────────────
def adjusted_r2(r2: float, n: int, p: int) -> float | str:
    """公式：1 - (1 - R²) × (n-1) / (n-p-1)"""
    denom = n - p - 1
    if denom <= 0:
        return "N/A（樣本數不足）"
    return 1 - (1 - r2) * (n - 1) / denom
