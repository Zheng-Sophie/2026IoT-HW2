"""
app.py — RegressionLab 主入口：Streamlit 布局、導航、Dataset Source 面板
"""
import streamlit as st
import sys
from pathlib import Path

# 確保本地模組可被匯入
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_DATASET_NAME, DEFAULT_TARGET_COLUMN,
    DEFAULT_SAMPLE_SIZE, DEFAULT_RANDOM_SEED, DEFAULT_TRAIN_RATIO,
)
from data_loader import (
    load_default_dataset, load_from_url, load_from_uploaded_file,
    clear_derived_results, dataset_fingerprint, get_dataset_load_instructions,
)
from data_validation import validate_dataframe, validate_target_column, get_candidate_features

# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RegressionLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全域 CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base:    #0f1117;
    --bg-surface: #1a1d2e;
    --bg-overlay: #1e2030;
    --border:     #313244;
    --text-main:  #cdd6f4;
    --text-muted: #6c7086;
    --accent-blue:#89b4fa;
    --accent-grn: #a6e3a1;
    --accent-red: #f38ba8;
    --accent-ylw: #f9e2af;
    --accent-mauve:#cba6f7;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text-main); }
.block-container { padding-top: 1.5rem; }
.dataset-panel { background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.metric-card { background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.8rem 1rem; text-align: center; }
.status-ok   { color: var(--accent-grn); font-weight: 600; }
.status-warn { color: var(--accent-ylw); font-weight: 600; }
.status-err  { color: var(--accent-red); font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600; background: var(--bg-overlay);
    color: var(--accent-blue); border: 1px solid var(--border); margin-right: 4px; }
h1, h2, h3 { color: var(--accent-blue); }
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ── Session State 初始化 ──────────────────────────────────────────────────────
def init_session():
    defaults = {
        "df": None,
        "df_fingerprint": None,
        "dataset_name": DEFAULT_DATASET_NAME,
        "dataset_source": "default",
        "dataset_status": "未載入",
        "target_col": DEFAULT_TARGET_COLUMN,
        "exclude_cols": [],
        "feature_cols": [],
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "random_seed": DEFAULT_RANDOM_SEED,
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "processed_data": None,
        "feature_selection_results": {},
        "experiment_results": None,
        "trained_models": {},
        "evaluation_results": {},
        "diagnostics_results": {},
        "best_model_key": None,
        "prediction_pipeline": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 RegressionLab")
    st.markdown("多元線性回歸分析平台")
    st.markdown("---")

    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"""
<div class='metric-card'>
<small class='text-muted'>資料集</small><br>
<b>{st.session_state.dataset_name}</b><br>
<small>{len(df_info):,} 列 × {df_info.shape[1]} 欄</small><br>
<small>目標：<code>{st.session_state.target_col}</code></small>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("尚未載入資料集")

    st.markdown("---")
    st.caption("**CRISP-DM 分析流程**")


# ── 右上方 Dataset Source 面板 ────────────────────────────────────────────────
def render_dataset_panel():
    with st.expander("📁 Dataset Source 設定", expanded=(st.session_state.df is None)):
        tab_default, tab_url, tab_upload = st.tabs(
            ["📌 預設資料集", "🔗 自訂 URL", "📤 上傳 CSV"]
        )

        # ── 預設資料集
        with tab_default:
            st.markdown(f"**{DEFAULT_DATASET_NAME}**")
            st.caption("Kaggle Student Performance Factors 資料集")
            if st.button("載入預設資料集", key="btn_load_default", type="primary"):
                with st.spinner("載入中..."):
                    df, msg = load_default_dataset()
                    if df is not None:
                        _set_dataset(df, DEFAULT_DATASET_NAME, "default", msg)
                    else:
                        st.error(msg)
                        st.info(get_dataset_load_instructions())

        # ── 自訂 URL
        with tab_url:
            url_input = st.text_input(
                "輸入公開 CSV 的直接下載網址",
                placeholder="https://raw.githubusercontent.com/.../data.csv",
                key="custom_url_input",
            )
            st.caption("⚠️ 僅支援可直接存取的 CSV URL（不支援需要登入的頁面）")
            if st.button("載入自訂 URL", key="btn_load_url", type="primary"):
                if not url_input.strip():
                    st.warning("請先輸入網址")
                else:
                    with st.spinner(f"下載中：{url_input[:60]}..."):
                        df, msg = load_from_url(url_input.strip())
                        if df is not None:
                            name = url_input.strip().split("/")[-1]
                            _set_dataset(df, name, "url", msg)
                        else:
                            st.error(msg)

        # ── 上傳 CSV
        with tab_upload:
            uploaded = st.file_uploader("上傳 CSV 檔案", type=["csv"], key="csv_uploader")
            if uploaded is not None:
                if st.button("確認載入上傳檔案", key="btn_load_upload", type="primary"):
                    with st.spinner("解析中..."):
                        df, msg = load_from_uploaded_file(uploaded)
                        if df is not None:
                            _set_dataset(df, uploaded.name, "upload", msg)
                        else:
                            st.error(msg)

        # ── 狀態顯示
        st.markdown("---")
        if st.session_state.df is not None:
            st.markdown(
                f"<span class='status-ok'>✅ 目前使用：{st.session_state.dataset_name}</span> "
                f"<span class='badge'>{len(st.session_state.df):,} 列</span>"
                f"<span class='badge'>{st.session_state.df.shape[1]} 欄</span>",
                unsafe_allow_html=True,
            )

            # 目標欄位選擇
            st.markdown("**目標欄位（Target Variable）**")
            numeric_cols = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
            default_idx = 0
            if st.session_state.target_col in numeric_cols:
                default_idx = numeric_cols.index(st.session_state.target_col)
            selected_target = st.selectbox(
                "選擇回歸目標欄位", numeric_cols,
                index=default_idx, key="target_col_selector"
            )
            if selected_target != st.session_state.target_col:
                st.session_state.target_col = selected_target
                # 清除依賴目標欄位的結果
                clear_derived_results()
                st.rerun()

            # 排除欄位
            st.markdown("**排除欄位（ID / 不適用特徵）**")
            all_cols = [c for c in st.session_state.df.columns
                       if c != st.session_state.target_col]
            exclude = st.multiselect(
                "選擇要排除的欄位", all_cols,
                default=st.session_state.exclude_cols,
                key="exclude_cols_selector",
            )
            if sorted(exclude) != sorted(st.session_state.exclude_cols):
                st.session_state.exclude_cols = exclude
                clear_derived_results()
                st.rerun()

            # 更新候選特徵
            st.session_state.feature_cols = get_candidate_features(
                st.session_state.df,
                st.session_state.target_col,
                st.session_state.exclude_cols,
            )
            st.info(f"候選特徵數：{len(st.session_state.feature_cols)} 個")
        else:
            st.warning("尚未載入任何資料集")


def _set_dataset(df, name, source, msg):
    """更新 session state 並清除舊結果。"""
    new_fp = dataset_fingerprint(df)
    if new_fp != st.session_state.df_fingerprint:
        clear_derived_results()
        st.session_state.exclude_cols = []  # reset exclusions for new dataset

    st.session_state.df = df
    st.session_state.df_fingerprint = new_fp
    st.session_state.dataset_name = name
    st.session_state.dataset_source = source

    # 動態選擇目標欄位：預設資料集優先用 DEFAULT_TARGET_COLUMN，否則讓使用者選
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if source == "default" and DEFAULT_TARGET_COLUMN in numeric_cols:
        st.session_state.target_col = DEFAULT_TARGET_COLUMN
    elif st.session_state.get("target_col", "") in numeric_cols:
        pass  # 保留上次選擇（若在新資料集中仍存在）
    else:
        # 自動選第一個數值欄位，讓使用者在面板確認
        st.session_state.target_col = numeric_cols[0] if numeric_cols else ""

    st.session_state.feature_cols = get_candidate_features(
        df, st.session_state.target_col, st.session_state.exclude_cols
    )
    st.success(msg)
    st.rerun()


# ── 主頁面 ────────────────────────────────────────────────────────────────────
st.title("📊 RegressionLab")
st.markdown("**多元線性回歸（Multiple Linear Regression）互動分析平台**")

render_dataset_panel()

st.markdown("---")

# 空狀態提示
if st.session_state.df is None:
    st.info("""
    👈 **開始分析前，請先從左上方的「Dataset Source 設定」載入資料集。**

    **預設資料集：** Student Performance Factors（Kaggle）
    - 目標變數：`Exam_Score`（指數成績）
    - CSV 檔名：`StudentPerformanceFactors.csv`
    - 包含數値欄位與類別欄位，系統自動處理 One-Hot Encoding
    - 若尚未下載，請使用上傳 CSV 功能或輸入可直接存取的 CSV 網址。

    **分析流程（CRISP-DM）：**
    使用左側導航列（Sidebar）依序瀏覽各頁面進行分析。
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h3>📌 Business Understanding</h3>
        <p>了解研究目標與問題背景</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h3>🔍 Data Understanding</h3>
        <p>探索資料結構與品質</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
        <h3>⚙️ Feature Selection</h3>
        <p>五種方法比較最佳特徵數</p>
        </div>""", unsafe_allow_html=True)
else:
    df = st.session_state.df
    st.subheader("📋 Overview — 目前狀態")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("資料集", st.session_state.dataset_name[:20])
    with c2:
        st.metric("資料列數", f"{len(df):,}")
    with c3:
        st.metric("候選特徵數", len(st.session_state.feature_cols))
    with c4:
        st.metric("目標欄位", st.session_state.target_col)

    # 快速摘要
    exp_status = "✅ 已完成" if st.session_state.experiment_results else "⏳ 尚未執行"
    model_status = "✅ 已訓練" if st.session_state.trained_models else "⏳ 尚未訓練"
    st.markdown(f"""
    | 狀態 | 說明 |
    |---|---|
    | 特徵選擇實驗 | {exp_status} |
    | 模型訓練 | {model_status} |
    | 評估 | {"✅ 已完成" if st.session_state.evaluation_results else "⏳ 尚未執行"} |
    | 診斷 | {"✅ 已完成" if st.session_state.diagnostics_results else "⏳ 尚未執行"} |
    """)
    st.info("👈 使用左側導航列選擇分析頁面。")
