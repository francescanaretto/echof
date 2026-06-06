#!/usr/bin/env python3
"""
Aggregate plot for the DP-query sweep experiments.

We read reports saved as:
  Reports-eval/<dataset>/report_original_on_real_test_logit_nn_25_dpquery_<eps>_laplace_True_wise.json

and produce an aggregate (mean ± std across datasets) plot for the selected eps values.

All configuration stays inside the script (no CLI args) for reproducibility on servers.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[2] / ".mplconfig").resolve())
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# CONFIG
ROOT = Path(__file__).resolve().parents[2]
REPORTS_EVAL_DIR = ROOT / "Reports-eval"
OUT_DIR = ROOT / "Reports-eval" / "dpquery_sweep"

GUIDING_BB = "nn"
KIND = "logit"
PERCENTILE = "25"

SPLIT = "real_test"  # "real_test" or "real_train" or "synth_test"
MECHANISM = "laplace"  # "laplace" or "gaussian"
WISE_ONLY = True

EPS_VALUES = [0.1, 0.5, 1.0, 5.0]

# Optional dataset filter. Leave empty to use all datasets that have the reports.
DATASETS_TO_USE: list[str] = []

BASE_FONT_SIZE = 16
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
    }
)

BAR_COLOR = "#CC79A7"   # pink
TEXT_COLOR = "#3F3F3F"  # dark gray
GRID_COLOR = "#D9D9D9"

YMIN, YMAX = 0.0, 1.0


REPORT_RE = re.compile(
    r"^report_original_on_(?P<split>real_test|real_train|synth_test)"
    r"_(?P<kind>.+?)_(?P<guiding_bb>.+?)_(?P<percentile>\d+)"
    r"_dpquery_(?P<eps>[0-9.]+)_(?P<mech>laplace|gaussian)"
    r"(?P<rest>.*)\.json$"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_report_dict(path: Path) -> dict | None:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    pos = txt.find("{")
    if pos == -1:
        return None
    try:
        return json.loads(txt[pos:])
    except Exception:
        return None


def extract_macro_f1(d: dict) -> float | None:
    if not isinstance(d, dict):
        return None
    macro = d.get("macro avg")
    if isinstance(macro, dict):
        v = macro.get("f1-score")
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


def load_dpquery_results() -> pd.DataFrame:
    eps_set = {str(e) for e in EPS_VALUES}
    rows: list[dict] = []

    for dataset_dir in REPORTS_EVAL_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        if DATASETS_TO_USE and dataset not in set(DATASETS_TO_USE):
            continue

        for path in dataset_dir.glob("report_original_on_*.json"):
            m = REPORT_RE.match(path.name)
            if not m:
                continue
            meta = m.groupdict()
            if meta["split"] != SPLIT:
                continue
            if meta["kind"] != KIND:
                continue
            if meta["guiding_bb"] != GUIDING_BB:
                continue
            if meta["percentile"] != PERCENTILE:
                continue
            if meta["mech"] != MECHANISM:
                continue
            if str(meta["eps"]) not in eps_set:
                continue
            if WISE_ONLY and "_wise" not in (meta.get("rest") or ""):
                continue

            d = load_report_dict(path)
            if d is None:
                continue
            f1 = extract_macro_f1(d)
            if f1 is None:
                continue
            rows.append({"dataset": dataset, "eps": float(meta["eps"]), "macro_f1": float(f1)})

    if not rows:
        raise ValueError("No dpquery reports found with the configured filters.")
    df = pd.DataFrame(rows)
    df["eps"] = df["eps"].astype(float)
    return df


def plot_aggregate(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)

    agg = (
        df.groupby("eps")["macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_datasets"})
    )
    agg = agg.sort_values("eps", ascending=True).reset_index(drop=True)
    agg.to_csv(OUT_DIR / "dpquery_sweep_summary.csv", index=False)

    x = np.arange(len(agg))
    means = agg["mean"].to_numpy(dtype=float)
    stds = agg["std"].fillna(0.0).to_numpy(dtype=float)

    fig_w = max(8.0, 2.0 + 1.6 * len(agg))
    fig_h = 4.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bars = ax.bar(x, means, color=BAR_COLOR, edgecolor="white", linewidth=1.2, zorder=2)
    ax.set_ylim(YMIN, YMAX)
    ax.set_ylabel("Fidelity Macro F1 (vs BB)", color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in agg["eps"].tolist()], color=TEXT_COLOR)
    ax.set_xlabel("DP Query Epsilon", color=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1.0, alpha=0.8, zorder=1)

    for rect, mean, std, n in zip(bars, means, stds, agg["n_datasets"].to_numpy(dtype=int)):
        msg = f"{mean:.2f} ± {std:.2f}"
        y_text = min(YMAX - 0.01, mean + 0.03)
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y_text,
            msg,
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=BASE_FONT_SIZE,
        )
        # Small n annotation at the base (helps interpret robustness).
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=max(10, BASE_FONT_SIZE - 4),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)

    out_png = OUT_DIR / f"dpquery_sweep_{KIND}_{GUIDING_BB}_{PERCENTILE}_{MECHANISM}{'_wise' if WISE_ONLY else ''}.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


def main() -> None:
    df = load_dpquery_results()
    # Persist full per-dataset table for debugging and paper appendix.
    ensure_dir(OUT_DIR)
    df.to_csv(OUT_DIR / "dpquery_sweep_per_dataset.csv", index=False)
    plot_aggregate(df)


if __name__ == "__main__":
    main()

