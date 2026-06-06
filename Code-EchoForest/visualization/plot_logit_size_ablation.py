#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the size ablation results stored in:
    Reports-eval/logit_nn_25_size_ablation_rf_summary.csv

Outputs:
1. One bar chart per dataset:
   x = increasing synthetic subset size
   y = F1 macro
2. One aggregate chart across datasets:
   x = increasing synthetic subset size
   y = mean F1 macro
   shaded band = +/- 1 std
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_FONT_SIZE = 16
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 1,
        "ytick.labelsize": BASE_FONT_SIZE - 1,
        "legend.fontsize": BASE_FONT_SIZE - 1,
    }
)


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "Reports-eval" / "size-ablation-logit-nn-25" / "logit_nn_25_size_ablation_rf_summary.csv"
OUT_DIR = ROOT / "Reports-eval" / "size-ablation-logit-nn-25" / "size-ablation-logit-nn-25-plots"

# Default metric. You can switch to "f1_original_test" if needed.
METRIC_COL = "f1_original_all"
METRIC_LABEL = "F1 Macro"

# Pastel, reasonably color-blind friendly palette from ColorBrewer Set2.
PALETTE = [
    "#66C2A5",  # teal
    "#FC8D62",  # soft orange
    "#8DA0CB",  # muted blue
    "#E78AC3",  # pink
    "#A6D854",  # light green
    "#FFD92F",  # soft yellow
]

AGG_LINE_COLOR = "#6D5BD0"
AGG_FILL_COLOR = "#AFC6E9"
GRID_COLOR = "#D9D9D9"
AGG_STD_FILL_COLOR = "#D9A8F2"
AGG_MARKER_COLOR = "#CC79A7"
AGG_MARKER_SIZE = 11

TITLE_SUFFIX = "Logit / NN / 25th percentile"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV non trovato: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required_cols = {"dataset", "subset_size", METRIC_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    df["subset_size"] = pd.to_numeric(df["subset_size"], errors="raise").astype(int)
    df[METRIC_COL] = pd.to_numeric(df[METRIC_COL], errors="raise").astype(float)

    df = df.sort_values(["dataset", "subset_size"]).reset_index(drop=True)
    return df


def format_subset_labels(values: list[int]) -> list[str]:
    return [f"{value:,}".replace(",", " ") for value in values]


def plot_dataset_bars(df_dataset: pd.DataFrame, out_dir: Path) -> None:
    dataset = df_dataset["dataset"].iloc[0]
    subset_sizes = df_dataset["subset_size"].tolist()
    scores = df_dataset[METRIC_COL].tolist()
    x = np.arange(len(subset_sizes))

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    colors = PALETTE[:len(subset_sizes)]

    bars = ax.bar(
        x,
        scores,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax.set_title(f"{dataset} | {TITLE_SUFFIX}", fontsize=BASE_FONT_SIZE, pad=12)
    ax.set_xlabel("Synthetic Subset Size", fontsize=BASE_FONT_SIZE)
    ax.set_ylabel(METRIC_LABEL, fontsize=BASE_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(format_subset_labels(subset_sizes), rotation=20, ha="right", fontsize=BASE_FONT_SIZE - 1)
    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE - 1)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

    y_min = max(0.0, min(scores) - 0.03)
    y_max = min(1.0, max(scores) + 0.03)
    ax.set_ylim(y_min, y_max)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.003,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=BASE_FONT_SIZE - 2,
            color="#3B3B3B",
        )

    out_path = out_dir / f"{dataset}_subset_bars_{METRIC_COL}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_mean_std(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby("subset_size", as_index=False)[METRIC_COL]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std"})
        .sort_values("subset_size")
        .reset_index(drop=True)
    )

    subset_sizes = summary["subset_size"].to_numpy(dtype=int)
    means = summary["f1_mean"].to_numpy(dtype=float)
    stds = summary["f1_std"].fillna(0.0).to_numpy(dtype=float)
    x = np.arange(len(subset_sizes))

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)

    ax.fill_between(
        x,
        np.clip(means - stds, 0.0, 1.0),
        np.clip(means + stds, 0.0, 1.0),
        color=AGG_STD_FILL_COLOR,
        alpha=0.32,
        zorder=1,
        label="Mean ± 1 std",
    )

    ax.plot(
        x,
        means,
        color=AGG_LINE_COLOR,
        linewidth=2.0,
        zorder=3,
        label="Mean F1",
    )

    ax.scatter(
        x,
        means,
        color=AGG_MARKER_COLOR,
        s=95,
        zorder=4,
    )

    ax.set_title(f"Average F-1 score across datasets ", fontsize=BASE_FONT_SIZE, pad=12)
    ax.set_xlabel("Synthetic Subset Size", fontsize=BASE_FONT_SIZE)
    ax.set_ylabel(METRIC_LABEL, fontsize=BASE_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(format_subset_labels(subset_sizes), rotation=20, ha="right", fontsize=BASE_FONT_SIZE - 1)
    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE - 1)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best", fontsize=BASE_FONT_SIZE - 1)

    y_min = max(0.0, float(np.min(means - stds)) - 0.03)
    y_max = min(1.0, float(np.max(means + stds)) + 0.03)
    ax.set_ylim(y_min, y_max)

    for xi, mean_value in zip(x, means):
        ax.text(
            xi,
            mean_value + 0.004,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=BASE_FONT_SIZE - 2,
            color="#3B3B3B",
        )

    out_path = out_dir / f"aggregate_mean_std_{METRIC_COL}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(OUT_DIR)
    df = load_data()

    for dataset, df_dataset in df.groupby("dataset", sort=True):
        plot_dataset_bars(df_dataset.sort_values("subset_size"), OUT_DIR)

    plot_aggregate_mean_std(df, OUT_DIR)
    print(f"[done] Plot saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
