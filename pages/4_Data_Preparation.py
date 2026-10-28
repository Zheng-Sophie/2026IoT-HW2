"""
pages/4_Data_Preparation.py — 資料準備頁（支援混合型資料 + ColumnTransformer）
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_SAMPLE_SIZE_OPTIONS, DEFAULT_SAMPLE_SIZE, DEFAULT_RANDOM_SEED, DEFAULT_TRAIN_RATIO
from data_preprocessing import (
    sample_data, split_data, preprocess_split,
    get_processed_data_summary, get_column_types,
)
from data_validation import validate_target_column

st.set_page_config(page_title="Data Preparation | RegressionLab", layout="wide")
st.title("⚙️ Data Preparation — 資料準備")

if st.session_state.get("df") is None:
    st.info("請先載入資料集。")
    st.stop()

df = st.session_state.df
target = st.session_state.get("target_col", "")
feat_cols = st.session_state.get("feature_cols", [])

if not feat_cols:
    st.warning("候選特徵為空，請返回首頁調整排除欄位設定。")
    st.stop()

ok, reason = validate_target_column(df, target)
if not ok:
    st.error(f"目標欄位問題：{reason}")
    st.stop()

# 顯示欄位類型資訊
X_preview = df[feat_cols]
num_cols_preview, cat_cols_preview = get_column_types(X_preview)

st.markdown("""
> **注意：** 此頁面進行的抽樣與切分設定，是後續所有特徵選擇、模型訓練與評估的基礎。
> 更改設定後，需重新執行「準備資料」以使後續分析與此設定一致。
""")

st.markdown(f"""
**特徵組成：**
- 數值欄位：{len(num_cols_preview)} 個
- 類別欄位（將 One-Hot Encoding）：{len(cat_cols_preview)} 個
  {('— ' + '、'.join(cat_cols_preview)) if cat_cols_preview else '（無）'}
""")

# ── 設定區 ────────────────────────────────────────────────────────────────────
st.subheader("1. 抽樣與切分設定")
c1, c2, c3 = st.columns(3)
with c1:
    sample_opts = DEFAULT_SAMPLE_SIZE_OPTIONS.copy()
    sample_opts_labels = [str(s) if isinstance(s, int) else s for s in sample_opts]
    cur_sample = st.session_state.get("sample_size", DEFAULT_SAMPLE_SIZE)
    cur_idx = 2  # 預設 50000
    for i, opt in enumerate(sample_opts):
        if opt == cur_sample:
            cur_idx = i
            break
    selected_label = st.selectbox("抽樣數量", sample_opts_labels, index=cur_idx, key="sample_size_sel")
    selected_sample = sample_opts[sample_opts_labels.index(selected_label)]

with c2:
    train_ratio = st.slider("訓練資料比例", 0.6, 0.9,
                            st.session_state.get("train_ratio", DEFAULT_TRAIN_RATIO), 0.05, key="train_ratio_sel")

with c3:
    rand_seed = st.number_input("隨機種子", 0, 9999,
                                st.session_state.get("random_seed", DEFAULT_RANDOM_SEED), key="rand_seed_sel")

n_avail = len(df)
n_sample = n_avail if selected_sample == "全部資料" else min(int(selected_sample), n_avail)
n_train = int(n_sample * train_ratio)
n_test = n_sample - n_train

st.info(f"**預計切分結果：** 抽樣 {n_sample:,} 筆 → 訓練集 {n_train:,} 筆 / 測試集 {n_test:,} 筆（資料集共 {n_avail:,} 筆）")
if n_sample < n_avail:
    st.caption("⚠️ 抽樣後的資料代表性有限，結果可能與使用全部資料有所差異。")

# ── 執行按鈕 ──────────────────────────────────────────────────────────────────
st.subheader("2. 執行資料準備")
run_btn = st.button("🚀 準備資料", type="primary", key="btn_prepare")

if run_btn:
    with st.spinner("抽樣、切分與前處理中..."):
        st.session_state["sample_size"] = selected_sample
        st.session_state["train_ratio"] = train_ratio
        st.session_state["random_seed"] = rand_seed

        # 清除依賴舊切分的結果
        for k in ["experiment_results", "trained_models", "evaluation_results",
                  "diagnostics_results", "best_model_key", "prediction_pipeline"]:
            if k in st.session_state:
                del st.session_state[k]

        sampled_df = sample_data(df, selected_sample, rand_seed)
        X_train_raw, X_test_raw, y_train, y_test = split_data(
            sampled_df, feat_cols, target, train_ratio, rand_seed
        )

        # 混合型前處理（ColumnTransformer，只在訓練集 fit）
        X_train_enc, X_test_enc, preprocessor, num_cols, cat_cols = preprocess_split(
            X_train_raw, X_test_raw
        )
        summary = get_processed_data_summary(
            X_train_enc, X_test_enc, y_train, y_test,
            orig_numeric_cols=num_cols, orig_categorical_cols=cat_cols,
        )

        st.session_state["processed_data"] = {
            "X_train": X_train_enc,       # 編碼後的 DataFrame
            "X_test": X_test_enc,
            "X_train_raw": X_train_raw,   # 原始（未編碼）— 供 Deployment 頁用
            "X_test_raw": X_test_raw,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessor": preprocessor,
            "summary": summary,
            "feature_names": X_train_enc.columns.tolist(),   # 編碼後特徵名
            "orig_feature_names": feat_cols,                  # 原始特徵名
            "orig_numeric_cols": num_cols,
            "orig_categorical_cols": cat_cols,
        }
    st.success("✅ 資料準備完成！")

# ── 結果顯示 ──────────────────────────────────────────────────────────────────
pd_data = st.session_state.get("processed_data")
if pd_data:
    st.subheader("3. 準備結果")
    s = pd_data["summary"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("訓練集筆數", f"{s['train_samples']:,}")
    with c2: st.metric("測試集筆數", f"{s['test_samples']:,}")
    with c3: st.metric("編碼後特徵數", s["n_features"])
    with c4: st.metric("目標欄位", target)

    c5, c6, c7, c8 = st.columns(4)
    with c5: st.metric("原始數值特徵數", len(s.get("orig_numeric_cols", [])))
    with c6: st.metric("原始類別特徵數（OHE 前）", len(s.get("orig_categorical_cols", [])))
    with c7: st.metric("訓練集目標均值", f"{s['train_target_mean']:.4f}")
    with c8: st.metric("測試集目標均值", f"{s['test_target_mean']:.4f}")

    st.markdown("---")
    st.subheader("4. 前處理說明")
    st.markdown("""
| 步驟 | 數值欄位 | 類別欄位 |
|---|---|---|
| 缺失值填補 | 中位數（fit on train only） | 眾數（fit on train only） |
| 編碼 / 縮放 | StandardScaler（fit on train only） | OneHotEncoder（fit on train only） |
| 測試集 | 使用訓練集的 scaler transform | 使用訓練集的 OHE transform |
| 未知類別 | — | `handle_unknown='ignore'`（補零） |

> ✅ 所有前處理只在訓練集 fit，測試集只 transform，確保無資料洩漏。
    """)

    st.subheader("5. 編碼後特徵清單")
    enc_feat_df = pd.DataFrame({
        "編碼後特徵名": pd_data["feature_names"],
        "來源": [
            "數值" if f in s.get("orig_numeric_cols", []) else "OHE"
            for f in pd_data["feature_names"]
        ]
    })
    enc_feat_df.index = enc_feat_df.index + 1
    st.dataframe(enc_feat_df, use_container_width=True)
