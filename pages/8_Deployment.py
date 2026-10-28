"""
pages/8_Deployment.py — 預測操作頁（支援混合型欄位：數字輸入框 + 類別下拉選單）
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_NAMES
from prediction_intervals import compute_prediction_interval, check_out_of_range

st.set_page_config(page_title="Deployment | RegressionLab", layout="wide")
st.title("🚀 Deployment — 預測操作")

if st.session_state.get("df") is None:
    st.info("請先載入資料集。")
    st.stop()

pred_pipeline_info = st.session_state.get("prediction_pipeline")
if pred_pipeline_info is None:
    st.warning("請先至「Modeling」頁面訓練至少一個模型。")
    st.stop()

pipeline     = pred_pipeline_info["pipeline"]
feat_names   = pred_pipeline_info["feature_names"]   # 編碼後特徵名
model_type   = pred_pipeline_info["model_type"]
model_label  = pred_pipeline_info["model_label"]
X_train      = pred_pipeline_info["X_train"]          # 編碼後的訓練 X
y_train      = pred_pipeline_info["y_train"]

# 取原始特徵資訊（供類別欄位輸入用）
pd_data = st.session_state.get("processed_data", {})
orig_feat_names   = pd_data.get("orig_feature_names", feat_names)
orig_numeric_cols = pd_data.get("orig_numeric_cols", feat_names)
orig_cat_cols     = pd_data.get("orig_categorical_cols", [])
preprocessor      = pd_data.get("preprocessor")       # ColumnTransformer

st.info(f"""
**目前採用模型：** {model_label}
**原始特徵數：** {len(orig_feat_names)} 個（數值 {len(orig_numeric_cols)} + 類別 {len(orig_cat_cols)}）
**編碼後特徵數：** {len(feat_names)} 個
**資料集：** {st.session_state.get('dataset_name', '—')}
""")
st.caption("⚠️ 切換資料集後，需重新訓練模型才能使用正確的預測 Pipeline。")
st.markdown("---")

# 取得類別欄位的合法值（從 df 抽取）
df_ref = st.session_state.df
cat_options: dict[str, list] = {}
for c in orig_cat_cols:
    if c in df_ref.columns:
        cat_options[c] = sorted(df_ref[c].dropna().unique().tolist())

# ────────────────────────────────────────────────────────────────────────────
# 用原始特徵建立輸入 → 透過 preprocessor 編碼 → pipeline predict
# ────────────────────────────────────────────────────────────────────────────
def _raw_input_to_encoded(raw_input: dict) -> pd.DataFrame:
    """將原始特徵 dict 轉成編碼後的 DataFrame，供 pipeline 預測。"""
    X_raw_df = pd.DataFrame([raw_input])
    if preprocessor is not None:
        # 使用訓練時 fit 的 ColumnTransformer
        arr = preprocessor.transform(X_raw_df)
        # 重建特徵名（與訓練時一致）
        from data_preprocessing import get_encoded_feature_names, get_column_types
        enc_names = get_encoded_feature_names(preprocessor, orig_numeric_cols, orig_cat_cols)
        return pd.DataFrame(arr, columns=enc_names, dtype=float)
    else:
        # fallback：直接回傳（只適用純數值資料集）
        return X_raw_df[feat_names].astype(float)


tab_single, tab_batch = st.tabs(["📝 單筆預測", "📋 CSV 批次預測"])

# ════════════════════════════════════════════════════════════════════════════
# 單筆預測
# ════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("單筆手動輸入預測")
    st.caption("數值欄位請輸入數字；類別欄位請從下拉選單選擇（若有）。")

    # 數值欄位的訓練集統計（用原始 X_train_raw 或 X_train）
    X_train_raw = pd_data.get("X_train_raw")
    if X_train_raw is None:
        X_train_raw = df_ref  # fallback

    with st.form("single_pred_form"):
        raw_input_vals: dict = {}
        # 分兩排：數值 / 類別各一組
        if orig_numeric_cols:
            st.markdown("**數值特徵**")
            cols_per_row = 3
            rows = [orig_numeric_cols[i:i+cols_per_row] for i in range(0, len(orig_numeric_cols), cols_per_row)]
            for row_feats in rows:
                rcols = st.columns(len(row_feats))
                for col, feat in zip(rcols, row_feats):
                    with col:
                        if feat in X_train_raw.columns:
                            default_val = float(pd.to_numeric(X_train_raw[feat], errors="coerce").mean())
                        else:
                            default_val = 0.0
                        if np.isnan(default_val):
                            default_val = 0.0
                        raw_input_vals[feat] = st.number_input(
                            feat, value=round(default_val, 4), format="%.4f",
                            key=f"num_input_{feat}",
                        )

        if orig_cat_cols:
            st.markdown("**類別特徵**")
            cols_per_row = 3
            rows = [orig_cat_cols[i:i+cols_per_row] for i in range(0, len(orig_cat_cols), cols_per_row)]
            for row_feats in rows:
                rcols = st.columns(len(row_feats))
                for col, feat in zip(rcols, row_feats):
                    with col:
                        opts = cat_options.get(feat, [])
                        raw_input_vals[feat] = st.selectbox(
                            feat, opts, key=f"cat_input_{feat}",
                        )

        submit_single = st.form_submit_button("🔮 預測", type="primary")

    if submit_single:
        try:
            X_encoded = _raw_input_to_encoded(raw_input_vals)
            # 僅保留 pipeline 訓練時選中的特徵名
            X_for_pred = X_encoded[feat_names] if all(f in X_encoded.columns for f in feat_names) else X_encoded

            # 超出範圍警告（數值特徵）
            numeric_feat_names = [f for f in feat_names if f in (orig_numeric_cols or [])]
            if numeric_feat_names and X_for_pred is not None:
                oor_warns = check_out_of_range(X_for_pred, X_train, numeric_feat_names)
                for feat, warn_msg in oor_warns.items():
                    st.warning(f"⚠️ **{feat}**：{warn_msg}")

            y_pred_single = pipeline.predict(X_for_pred)[0]
            st.success(f"## 預測結果：**{y_pred_single:.4f}**")

            # 預測區間
            pi_info = compute_prediction_interval(
                pipeline, X_train, y_train, X_for_pred, feat_names, model_type
            )
            if pi_info.get("supported"):
                lo = pi_info["pi_lower"][0]
                hi = pi_info["pi_upper"][0]
                st.info(f"🎯 **95% 預測區間（個別觀測）：** [{lo:.4f}, {hi:.4f}]")
                st.caption(pi_info.get("note", ""))
            else:
                st.warning(f"⚠️ **預測區間：** {pi_info.get('note', '此模型不支援預測區間。')}")

            with st.expander("查看輸入特徵值"):
                st.dataframe(pd.DataFrame([raw_input_vals]).T.rename(columns={0: "輸入值"}),
                             use_container_width=True)
        except Exception as e:
            st.error(f"預測失敗：{e}")


# ════════════════════════════════════════════════════════════════════════════
# CSV 批次預測
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("CSV 批次預測")
    st.markdown(f"""
**必要欄位（原始特徵，共 {len(orig_feat_names)} 個）：**
`{'`、`'.join(orig_feat_names)}`

- 數值欄位請提供數字值
- 類別欄位請與訓練資料一致（大小寫敏感）
- **不需要**包含目標欄位（`{st.session_state.get('target_col', '')}`）；若有包含，系統可比較預測值與真實值。
    """)

    uploaded_batch = st.file_uploader("上傳 CSV 檔案（批次預測）", type=["csv"], key="batch_csv")

    if uploaded_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)
        except Exception as e:
            st.error(f"CSV 解析失敗：{e}")
            st.stop()

        st.write(f"✅ 已上傳 {len(batch_df):,} 筆資料，共 {batch_df.shape[1]} 個欄位。")
        target_col = st.session_state.get("target_col", "")

        # 欄位驗證（原始特徵）
        missing_cols = [f for f in orig_feat_names if f not in batch_df.columns]
        if missing_cols:
            st.error(f"❌ 缺少必要欄位：{missing_cols}")
            st.info("請確認 CSV 欄位名稱與訓練時的**原始**特徵一致。")
        else:
            extra_cols = [c for c in batch_df.columns if c not in orig_feat_names and c != target_col]
            if extra_cols:
                st.warning(f"⚠️ 以下欄位不在模型特徵中，將被忽略：{extra_cols}")

            # 純數值欄位型態檢查
            type_errors = [f for f in orig_numeric_cols if f in batch_df.columns
                           and not pd.api.types.is_numeric_dtype(batch_df[f])]
            if type_errors:
                st.error(f"❌ 以下數值欄位包含非數字資料：{type_errors}")
            else:
                # 缺失值摘要
                ms = batch_df[orig_feat_names].isnull().sum()
                ms = ms[ms > 0]
                if not ms.empty:
                    st.warning(f"⚠️ 以下欄位有缺失值（Pipeline 自動填補）：{ms.to_dict()}")

                btn_batch = st.button("🔮 執行批次預測", type="primary", key="btn_batch_pred")
                if btn_batch:
                    try:
                        X_batch_raw = batch_df[orig_feat_names].copy()
                        # 使用訓練時 fit 的 ColumnTransformer 編碼
                        if preprocessor is not None:
                            X_batch_arr = preprocessor.transform(X_batch_raw)
                            from data_preprocessing import get_encoded_feature_names
                            enc_names = get_encoded_feature_names(preprocessor, orig_numeric_cols, orig_cat_cols)
                            X_batch_enc = pd.DataFrame(X_batch_arr, columns=enc_names, dtype=float)
                        else:
                            X_batch_enc = X_batch_raw.astype(float)

                        X_batch_for_pred = X_batch_enc[feat_names] if all(f in X_batch_enc.columns for f in feat_names) else X_batch_enc
                        y_batch_pred = pipeline.predict(X_batch_for_pred)

                        result_df = batch_df.copy()
                        result_df["Predicted"] = y_batch_pred.round(4)

                        # 若 CSV 含真實目標值，加一欄誤差
                        if target_col and target_col in batch_df.columns:
                            y_true_batch = pd.to_numeric(batch_df[target_col], errors="coerce")
                            result_df["Error (Pred - True)"] = (y_batch_pred - y_true_batch).round(4)
                            st.success(f"✅ 批次預測完成，並與真實 `{target_col}` 比較！共 {len(result_df):,} 筆。")
                        else:
                            st.success(f"✅ 批次預測完成！共 {len(result_df):,} 筆。")

                        disp_cols = orig_feat_names[:4] + ["Predicted"]
                        if "Error (Pred - True)" in result_df.columns:
                            disp_cols += ["Error (Pred - True)"]
                        st.dataframe(result_df[disp_cols].head(50), use_container_width=True)
                        st.caption("顯示前 50 筆結果、前 4 個原始特徵欄位及預測值。")

                        st.subheader("批次結果摘要")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("預測筆數", f"{len(y_batch_pred):,}")
                        with c2: st.metric("預測值最小", f"{y_batch_pred.min():.4f}")
                        with c3: st.metric("預測值最大", f"{y_batch_pred.max():.4f}")
                        with c4: st.metric("預測值均值", f"{y_batch_pred.mean():.4f}")
                        st.info("ℹ️ 第一版不提供預測結果下載或 CSV 匯出功能。")
                    except Exception as e:
                        st.error(f"批次預測失敗：{e}")
