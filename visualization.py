"""
visualization.py — 所有圖表繪製（matplotlib / seaborn）
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 非互動式後端，適合 Streamlit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── 統一樣式 ──────────────────────────────────────────────────────────────────
PALETTE = "tab10"
FIG_DPI = 120

plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "SimHei"
]

plt.rcParams["axes.unicode_minus"] = False

def _new_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=FIG_DPI)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#cdd6f4")
    ax.xaxis.label.set_color("#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.title.set_color("#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")
    return fig, ax


def _style_ax(ax):
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#cdd6f4")
    ax.xaxis.label.set_color("#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.title.set_color("#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")
    return ax



# ── 特徵數曲線（並排）────────────────────────────────────────────────────────
def plot_feature_count_curves(
    n_features_list: list[int],
    rmse_means: list[float],
    r2_means: list[float],
    rmse_stds: list[float] | None = None,
    r2_stds: list[float] | None = None,
    title_suffix: str = "",
    show_errorbars: bool = True,
) -> plt.Figure:
    """
    並排顯示：
      左圖 — RMSE by Number of Features
      右圖 — R-squared by Number of Features
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        _style_ax(ax)

    xs = n_features_list
    color_rmse = "#89b4fa"
    color_r2   = "#a6e3a1"

    # --- RMSE ---
    if show_errorbars and rmse_stds:
        ax1.errorbar(xs, rmse_means, yerr=rmse_stds,
                     fmt="o-", color=color_rmse, ecolor="#6c7086",
                     capsize=4, linewidth=2, markersize=6, label="CV RMSE (mean ± std)")
    else:
        ax1.plot(xs, rmse_means, "o-", color=color_rmse, linewidth=2, markersize=6)
    ax1.set_xlabel("Number of Features", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title(f"RMSE by Number of Features\n{title_suffix}", fontsize=13)
    ax1.set_xticks(xs)
    ax1.grid(True, color="#313244", alpha=0.5)
    ax1.legend(fontsize=9, facecolor="#1e2030", labelcolor="#cdd6f4")

    # --- R² ---
    if show_errorbars and r2_stds:
        ax2.errorbar(xs, r2_means, yerr=r2_stds,
                     fmt="s-", color=color_r2, ecolor="#6c7086",
                     capsize=4, linewidth=2, markersize=6, label="CV R² (mean ± std)")
    else:
        ax2.plot(xs, r2_means, "s-", color=color_r2, linewidth=2, markersize=6)
    ax2.set_xlabel("Number of Features", fontsize=12)
    ax2.set_ylabel("R-squared", fontsize=12)
    ax2.set_title(f"R-squared by Number of Features\n{title_suffix}", fontsize=13)
    ax2.set_xticks(xs)
    ax2.grid(True, color="#313244", alpha=0.5)
    ax2.legend(fontsize=9, facecolor="#1e2030", labelcolor="#cdd6f4")

    fig.suptitle("Train set Cross-Validation resule(not Test Set)",
                 fontsize=11, color="#6c7086", y=1.01)
    fig.tight_layout()
    return fig


# ── Actual vs. Predicted ──────────────────────────────────────────────────────
def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted",
) -> plt.Figure:
    fig, ax = _new_fig(6, 6)
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#89b4fa", label="觀測點")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y = x (完美預測)")
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, facecolor="#1e2030", labelcolor="#cdd6f4")
    ax.grid(True, color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig


# ── Residuals vs. Fitted ──────────────────────────────────────────────────────
def plot_residuals_vs_fitted(
    y_fitted: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuals vs. Fitted",
) -> plt.Figure:
    fig, ax = _new_fig(7, 5)
    ax.scatter(y_fitted, residuals, alpha=0.4, s=15, color="#cba6f7", label="殘差")
    ax.axhline(0, color="#f38ba8", linewidth=1.5, linestyle="--", label="零殘差參考線")
    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, facecolor="#1e2030", labelcolor="#cdd6f4")
    ax.grid(True, color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig


# ── Q-Q Plot ──────────────────────────────────────────────────────────────────
def plot_qq(
    theoretical_q: np.ndarray,
    sample_q: np.ndarray,
    title: str = "Residual Q-Q Plot",
) -> plt.Figure:
    fig, ax = _new_fig(5, 5)
    ax.scatter(theoretical_q, sample_q, alpha=0.5, s=15, color="#fab387", label="殘差分位數")
    # 45° 參考線
    min_v = min(theoretical_q.min(), sample_q.min())
    max_v = max(theoretical_q.max(), sample_q.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Normal Q-Q line")
    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, facecolor="#1e2030", labelcolor="#cdd6f4")
    ax.grid(True, color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig


# ── 係數 / 重要性圖 ───────────────────────────────────────────────────────────
def plot_coefficients(
    feature_names: list[str],
    values: list[float],
    title: str = "Feature Coefficients",
    x_label: str = "係數值",
) -> plt.Figure:
    n = len(feature_names)
    fig, ax = _new_fig(8, max(4, n * 0.35 + 1))
    colors = ["#89b4fa" if v >= 0 else "#f38ba8" for v in values]
    sorted_pairs = sorted(zip(feature_names, values), key=lambda x: abs(x[1]))
    names_s = [p[0] for p in sorted_pairs]
    vals_s  = [p[1] for p in sorted_pairs]
    colors_s = ["#89b4fa" if v >= 0 else "#f38ba8" for v in vals_s]
    ax.barh(names_s, vals_s, color=colors_s, edgecolor="#313244")
    ax.axvline(0, color="#6c7086", linewidth=1)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="x", color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig


# ── 模型比較長條圖 ────────────────────────────────────────────────────────────
def plot_model_comparison(
    model_labels: list[str],
    rmse_values: list[float],
    r2_values: list[float],
    source: str = "Test Set",
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        _style_ax(ax)

    colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7"][:len(model_labels)]

    ax1.bar(model_labels, rmse_values, color=colors, edgecolor="#313244")
    ax1.set_title(f"RMSE 比較（{source}）", fontsize=13)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.grid(True, axis="y", color="#313244", alpha=0.5)

    ax2.bar(model_labels, r2_values, color=colors, edgecolor="#313244")
    ax2.set_title(f"R² 比較（{source}）", fontsize=13)
    ax2.set_ylabel("R²", fontsize=12)
    ax2.grid(True, axis="y", color="#313244", alpha=0.5)

    fig.suptitle("模型評估比較", fontsize=14, color="#cdd6f4")
    fig.tight_layout()
    return fig


# ── 相關性熱圖 ────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Feature Correlation Heatmap",
    max_features: int = 25,
) -> plt.Figure:
    # 若特徵過多，截取前 max_features 個
    cols = corr_matrix.columns[:max_features]
    corr = corr_matrix.loc[cols, cols]
    n = len(cols)
    fig_size = max(6, min(n * 0.55 + 1, 18))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85), dpi=FIG_DPI)
    fig.patch.set_facecolor("#0f1117")
    _style_ax(ax)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=(n <= 15), fmt=".2f", linewidths=0.3,
        linecolor="#313244",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8, "color": "#cdd6f4"},
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis="x", rotation=45, colors="#cdd6f4", labelsize=8)
    ax.tick_params(axis="y", rotation=0, colors="#cdd6f4", labelsize=8)
    fig.tight_layout()
    return fig


# ── 目標變數分布 ──────────────────────────────────────────────────────────────
def plot_target_distribution(
    y: pd.Series,
    target_name: str = "Target",
) -> plt.Figure:
    fig, ax = _new_fig(7, 4)
    sns.histplot(y, ax=ax, kde=True, color="#89b4fa", edgecolor="#313244", bins=50)
    ax.set_xlabel(target_name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{target_name} Distribution", fontsize=13)
    ax.grid(True, color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig


# ── 特徵重要性（Permutation 或 Score）────────────────────────────────────────
def plot_feature_importance(
    feature_names: list[str],
    scores: list[float],
    title: str = "Feature Importance",
    top_n: int = 20,
) -> plt.Figure:
    pairs = sorted(zip(feature_names, scores), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names_s = [p[0] for p in pairs]
    vals_s  = [p[1] for p in pairs]
    n = len(names_s)
    fig, ax = _new_fig(8, max(4, n * 0.4 + 1))
    colors = ["#89b4fa" if v >= 0 else "#f38ba8" for v in vals_s]
    ax.barh(names_s[::-1], vals_s[::-1], color=colors[::-1], edgecolor="#313244")
    ax.axvline(0, color="#6c7086", linewidth=1)
    ax.set_xlabel("Score / Importance", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="x", color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig


# ── 殘差分布直方圖 ────────────────────────────────────────────────────────────
def plot_residual_histogram(
    residuals: np.ndarray,
    title: str = "Residual Distribution",
) -> plt.Figure:
    fig, ax = _new_fig(6, 4)
    sns.histplot(residuals, ax=ax, kde=True, color="#cba6f7", edgecolor="#313244", bins=50)
    ax.axvline(0, color="#f38ba8", linewidth=1.5, linestyle="--", label="零殘差")
    ax.set_xlabel("Residuals", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, facecolor="#1e2030", labelcolor="#cdd6f4")
    ax.grid(True, color="#313244", alpha=0.5)
    fig.tight_layout()
    return fig
