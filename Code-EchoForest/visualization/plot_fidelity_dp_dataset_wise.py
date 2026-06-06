#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot DP-query sweep results for *fidelity* (student RF vs original BB).

Reads the summary CSV produced by:
  Code-anonymous/validation/validation_fidelity_dp.py

Outputs:
  Reports-eval/fidelity_dpquery_sweep/dataset_wise/
    - aggregated mean/std plot across datasets
    - per-dataset plot
    - corresponding CSVs

Example:
  DATASETS=activity,pol,spotify,spotify-r KIND=entropy GUIDING_BB=nn PERCENTILE=25 \
  QUERY_NOISE_MECH=laplace NOISE_ON_LABELING=True \
  python3 Code/visualization/plot_fidelity_dp_dataset_wise.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend and a writable Matplotlib cache so the script
# runs reliably on remote/sandboxed machines.
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[2] / ".mplconfig").resolve())
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_FONT_SIZE = 16
# Force *all* relevant font sizes to 16 (Matplotlib otherwise keeps separate defaults).
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "figure.titlesize": BASE_FONT_SIZE,
    }
)

PLOT_TITLE = "Differential Privacy during synthetic data generation"
Y_AXIS_MIN = 0.00
Y_AXIS_MAX = 1.00
# Pastel-ish tones on purple/pink (requested).
PALETTE = {
    "mean": "#6D5BD0",    # soft purple
    "band": "#D9A8F2",    # slightly darker lavender (std band)
    "points": "#CC79A7",  # pink/purple points
    "text": "#2F2F2F",
}
MEAN_MARKER_SIZE = 11


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_reports_dir_env = os.environ.get("REPORTS_DIR", "").strip()
if _reports_dir_env:
    _p = Path(_reports_dir_env)
    REPORTS_DIR = (_p if _p.is_absolute() else (PROJECT_ROOT / _p)).resolve()
else:
    REPORTS_DIR = (PROJECT_ROOT / "Reports-eval").resolve()


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_list_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


TARGET_KIND = os.environ.get("KIND", "entropy").strip()
TARGET_GUIDING_BB = os.environ.get("GUIDING_BB", "nn").strip()
TARGET_PERCENTILE = str(int(os.environ.get("PERCENTILE", "25").strip()))
TARGET_MECH = os.environ.get("QUERY_NOISE_MECH", "laplace").strip()
NOISE_ON_LABELING = parse_bool_env("NOISE_ON_LABELING", True)

SPLIT = os.environ.get("SPLIT", "real_test").strip()  # real_train | real_test
METRIC = os.environ.get("METRIC", "fidelity_macro_f1").strip()  # fidelity_macro_f1 | fidelity_accuracy

DP_ORDER = [
    item.strip()
    for item in os.environ.get("DP_ORDER", "0.1,0.5,1.0,5.0").split(",")
    if item.strip()
]
DP_ORDER_MAP = {eps: idx for idx, eps in enumerate(DP_ORDER)}

# If None -> use all datasets present
SELECT_DATASETS = parse_list_env("SELECT_DATASETS", ["california", "htru2"])
if not SELECT_DATASETS:
    SELECT_DATASETS = None


def _csv_path() -> Path:
    noise = "True" if NOISE_ON_LABELING else "False"
    return (
        REPORTS_DIR
        / "fidelity_dpquery_sweep"
        / f"fidelity_student_vs_bb_{TARGET_KIND}_{TARGET_GUIDING_BB}_{TARGET_PERCENTILE}_{TARGET_MECH}_{noise}.csv"
    )


def main() -> None:
    csv_path = _csv_path()
    print(f"[info] using script: {Path(__file__).resolve()}")
    print(f"[info] REPORTS_DIR={REPORTS_DIR}")
    print(f"[info] input CSV={csv_path}")
    print(f"[info] filters: kind={TARGET_KIND} guiding_bb={TARGET_GUIDING_BB} percentile={TARGET_PERCENTILE} mech={TARGET_MECH} noise_on_labeling={NOISE_ON_LABELING}")
    print(f"[info] split={SPLIT} metric={METRIC}")
    print(f"[info] SELECT_DATASETS={SELECT_DATASETS}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV non trovato: {csv_path}\n"
            "Run Code-anonymous/validation/validation_fidelity_dp.py to generate the summary CSV."
        )

    df = pd.read_csv(csv_path)

    # Normalize types
    df["dp_epsilon"] = df["dp_epsilon"].astype(str)
    df["percentile"] = df["percentile"].astype(str)
    df["noise_on_labeling"] = df["noise_on_labeling"].astype(bool)

    # Base filters
    df = df[
        (df["kind"].astype(str) == TARGET_KIND)
        & (df["guiding_bb"].astype(str) == TARGET_GUIDING_BB)
        & (df["percentile"].astype(str) == TARGET_PERCENTILE)
        & (df["mechanism"].astype(str) == TARGET_MECH)
        & (df["noise_on_labeling"] == bool(NOISE_ON_LABELING))
        & (df["split"].astype(str) == SPLIT)
    ].copy()

    if SELECT_DATASETS is not None:
        df = df[df["dataset"].isin(SELECT_DATASETS)].copy()

    if df.empty:
        raise RuntimeError("No data after filtering. Check SPLIT/METRIC and the configuration used in validation_fidelity_dp.py.")

    # Order eps
    df["dp_order"] = df["dp_epsilon"].map(DP_ORDER_MAP).fillna(10_000).astype(int)
    df = df.sort_values(["dp_order", "dp_epsilon", "dataset"])

    # Output dir
    out_dir = REPORTS_DIR / "fidelity_dpquery_sweep" / "dataset_wise"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregation across datasets
    agg = (
        df.groupby(["mechanism", "dp_epsilon", "dp_order"], as_index=False)
        .agg(
            mean=(METRIC, "mean"),
            std=(METRIC, "std"),
            count=(METRIC, "count"),
        )
    )
    agg = agg.sort_values(["dp_order", "dp_epsilon"])

    print("[info] dataset disponibili:", sorted(df["dataset"].unique().tolist()))
    print("[info] summary:")
    print(agg.to_string(index=False))

    # Save CSVs
    noise = "true" if NOISE_ON_LABELING else "false"
    base = f"{METRIC}_dpquery_{SPLIT}_{noise}_{TARGET_MECH}"
    out_csv = out_dir / f"{base}.csv"
    agg.to_csv(out_csv, index=False)
    print(f"[OK] CSV saved to: {out_csv}")

    # Plot aggregate: mean line + std band (like plot_dp_dataset_wise.py)
    xs = np.arange(len(agg))
    labels = agg["dp_epsilon"].tolist()
    means = agg["mean"].to_numpy()
    stds = agg["std"].fillna(0.0).to_numpy()
    upper = np.clip(means + stds, Y_AXIS_MIN, Y_AXIS_MAX)
    lower = np.clip(means - stds, Y_AXIS_MIN, Y_AXIS_MAX)

    plt.figure(figsize=(9, 5))
    # std band
    plt.fill_between(xs, lower, upper, color=PALETTE["band"], alpha=0.32, linewidth=0, label="±1 std")
    # mean
    plt.plot(
        xs,
        means,
        "-o",
        lw=2.8,
        color=PALETTE["mean"],
        markerfacecolor=PALETTE["points"],
        markersize=MEAN_MARKER_SIZE,
    )
    plt.xticks(xs, labels)
    plt.ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    plt.xlabel("DP epsilon")
    if METRIC == "fidelity_macro_f1":
        ylab = "Fidelity Macro F-1"
    elif METRIC == "fidelity_accuracy":
        ylab = "Fidelity Accuracy"
    else:
        ylab = METRIC.replace("_", " ")
    plt.ylabel(ylab)
    plt.title(PLOT_TITLE)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    out_png = out_dir / f"{base}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Plot saved to: {out_png}")

    # Per-dataset lines
    per = df.pivot_table(index="dp_epsilon", columns="dataset", values=METRIC, aggfunc="mean")
    # enforce dp order
    per = per.reindex(index=[e for e in DP_ORDER if e in per.index] + [e for e in per.index if e not in DP_ORDER])

    out_csv_per = out_dir / f"{base}_per_dataset.csv"
    per.to_csv(out_csv_per)
    print(f"[OK] CSV per-dataset saved to: {out_csv_per}")

    plt.figure(figsize=(10, 5))
    for ds in per.columns:
        plt.plot(per.index.tolist(), per[ds].to_numpy(), marker="o", lw=1.8, alpha=0.9, label=str(ds))
    plt.ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    plt.xlabel("DP epsilon")
    plt.ylabel(METRIC.replace("_", " "))
    plt.title(PLOT_TITLE + f" ({SPLIT})")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    out_png_per = out_dir / f"{base}_per_dataset.png"
    plt.savefig(out_png_per, dpi=160)
    plt.close()
    print(f"[OK] Plot per-dataset saved to: {out_png_per}")


if __name__ == "__main__":
    main()
