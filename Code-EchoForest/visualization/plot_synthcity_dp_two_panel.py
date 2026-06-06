#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-panel synthetic quality plot for DP-query runs (epsilon sweep).

This mirrors plots/performance-plots/synthetic_quality_two_panel.png, but
aggregates across datasets for each DP epsilon (mean ± std).

Inputs:
  JSON reports produced by Code/validation/synth_gen_validation.py in DPQUERY_MODE,
  stored under: Model-synthetic/performance/
  Example filename:
    htru2_nn_entropy_25_dpquery_0.1_laplace_True_False_minimal_sanity_checks.json

Metrics used (from payload["results"]):
  - "nearest_synthetic_neighbor_distance"  -> shown as "NN distance"
  - "xgb_performance"                     -> shown as "XGB synth"

Output:
  plots/performance-plots/synthetic_quality_two_panel_dpquery.png
  plots/performance-plots/synthetic_quality_two_panel_dpquery_summary.csv

Config via env vars (optional):
  KIND=entropy
  MODEL=nn
  PERCENTILE=25
  DPQUERY_MECH=laplace
  DPQUERY_NOISE_ON_LABELING=True
  DP_ORDER=0.1,0.5,1.0,5.0
  REPORTS_DIR=/abs/path/to/repo (rare; defaults to repo root)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Ensure Matplotlib uses a writable cache directory and a non-interactive backend.
ROOT = Path(__file__).resolve().parents[2]
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".mplconfig").resolve())
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t"}


KIND = os.environ.get("KIND", "entropy").strip()
MODEL = os.environ.get("MODEL", os.environ.get("GUIDING_BB", "nn")).strip()  # keep compat
PERCENTILE = str(int(os.environ.get("PERCENTILE", "25").strip()))
DPQUERY_MECH = os.environ.get("DPQUERY_MECH", os.environ.get("QUERY_NOISE_MECH", "laplace")).strip()
DPQUERY_NOISE_ON_LABELING = _parse_bool(os.environ.get("DPQUERY_NOISE_ON_LABELING", os.environ.get("NOISE_ON_LABELING", "True")))

DP_ORDER = [x.strip() for x in os.environ.get("DP_ORDER", "0.1,0.5,1.0,5.0").split(",") if x.strip()]
DP_ORDER_MAP = {eps: i for i, eps in enumerate(DP_ORDER)}

IN_DIR = (ROOT / "Model-synthetic" / "performance").resolve()
OUT_DIR = (ROOT / "plots" / "performance-plots").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG = OUT_DIR / "synthetic_quality_two_panel_dpquery.png"
OUT_CSV = OUT_DIR / "synthetic_quality_two_panel_dpquery_summary.csv"

FONT_SIZE = 16
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
})

# Pastel purple/pink palette
COLOR_DP = "#CC79A7"     # pink
EDGE_COLOR = "#4A4A4A"
GRID_COLOR = "#DDDDDD"

METRICS = [
    ("xgb_performance", "XGB synth", "XGB synth. performance"),
    ("nearest_synthetic_neighbor_distance", "NN distance", "Nearest neigh. separation"),
]


REPORT_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<model>nn|rf)_(?P<kind>.+?)_(?P<percentile>\d+)_dpquery_"
    r"(?P<eps>[0-9.]+)_(?P<mech>laplace|gaussian)_(?P<noise>True|False)_"
    r"(?P<standardize>True|False)_minimal_sanity_checks\.json$"
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric_value(metric_obj) -> float | None:
    """
    SynthCity metrics usually return a float, or a dict with 'mean'/'value' keys.
    We handle a few common shapes.
    """
    if metric_obj is None:
        return None
    if isinstance(metric_obj, (int, float, np.floating)):
        return float(metric_obj)
    if isinstance(metric_obj, dict):
        # If the metric failed, SynthCity often stores {"error": "..."}.
        if "error" in metric_obj:
            return None

        # SynthCity PerformanceEvaluatorXGB often returns:
        # {"gt": <score>, "syn_id": <score>, "syn_ood": <score>}
        # For "XGB synth" we want the synthetic-driven utility, i.e. "syn_id".
        if "syn_id" in metric_obj and isinstance(metric_obj.get("syn_id"), (int, float, np.floating)):
            return float(metric_obj["syn_id"])

        # common shapes:
        # - {"mean": 0.7, "std": 0.1}
        # - {"value": 0.7}
        # - {"mean": {"mean": 0.7, ...}, ...}
        for k in ("value", "mean", "score"):
            if k not in metric_obj:
                continue
            v = metric_obj.get(k)
            if isinstance(v, (int, float, np.floating)):
                return float(v)
            if isinstance(v, dict):
                nested = _extract_metric_value(v)
                if nested is not None:
                    return float(nested)

        # Fallback: recursively search for the first numeric leaf.
        def _search(obj, depth: int) -> float | None:
            if depth <= 0:
                return None
            if isinstance(obj, (int, float, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                if "error" in obj:
                    return None
                for vv in obj.values():
                    got = _search(vv, depth - 1)
                    if got is not None:
                        return got
            if isinstance(obj, (list, tuple)):
                for vv in obj:
                    got = _search(vv, depth - 1)
                    if got is not None:
                        return got
            return None

        return _search(metric_obj, depth=4)
    return None


def main() -> None:
    print(f"[info] using script: {Path(__file__).resolve()}")
    print(f"[info] IN_DIR={IN_DIR}")
    print(f"[info] OUT_DIR={OUT_DIR}")
    print(f"[info] filters: model={MODEL} kind={KIND} percentile={PERCENTILE} mech={DPQUERY_MECH} noise_on_labeling={DPQUERY_NOISE_ON_LABELING}")
    print(f"[info] DP_ORDER={DP_ORDER}")

    if not IN_DIR.exists():
        raise FileNotFoundError(f"Directory non trovata: {IN_DIR} (hai generato i report synth_gen_validation.py?)")

    rows: list[dict] = []
    for p in sorted(IN_DIR.glob("*_minimal_sanity_checks.json")):
        m = REPORT_RE.match(p.name)
        if not m:
            continue
        gd = m.groupdict()
        if gd["model"] != MODEL:
            continue
        if gd["kind"] != KIND:
            continue
        if gd["percentile"] != PERCENTILE:
            continue
        if gd["mech"] != DPQUERY_MECH:
            continue
        noise_flag = _parse_bool(gd["noise"])
        if noise_flag != bool(DPQUERY_NOISE_ON_LABELING):
            continue
        eps = gd["eps"]
        if eps not in DP_ORDER:
            continue

        payload = _load_json(p)
        results = payload.get("results", {})

        for metric_key, metric_name, _title in METRICS:
            val = _extract_metric_value(results.get(metric_key))
            if val is None:
                continue
            rows.append({
                "dataset": gd["dataset"],
                "method": f"dpquery eps={eps}",
                "dp_epsilon": eps,
                "metric": metric_name,
                "value": float(val),
                "file": str(p),
            })

    if not rows:
        raise ValueError("Nessun report DP trovato dopo i filtri. Check KIND/MODEL/PERCENTILE/DPQUERY_MECH/DPQUERY_NOISE_ON_LABELING/DP_ORDER.")

    df = pd.DataFrame(rows)
    df["dp_order"] = df["dp_epsilon"].map(DP_ORDER_MAP).astype(int)

    # Helpful debug: show which metrics are actually present after parsing.
    print(f"[info] parsed metrics: {sorted(df['metric'].unique().tolist())}")

    # aggregate across datasets
    agg = (
        df.groupby(["dp_epsilon", "dp_order", "metric"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
        .sort_values(["metric", "dp_order"])
        .reset_index(drop=True)
    )

    # wide for plotting
    wide_mean = agg.pivot_table(index=["dp_epsilon", "dp_order"], columns="metric", values="mean").reset_index()
    wide_std = agg.pivot_table(index=["dp_epsilon", "dp_order"], columns="metric", values="std").reset_index()
    wide = wide_mean.merge(wide_std, on=["dp_epsilon", "dp_order"], suffixes=("_mean", "_std"))
    wide = wide.sort_values("dp_order").reset_index(drop=True)

    wide.to_csv(OUT_CSV, index=False)
    print(f"[OK] summary: {OUT_CSV}")
    print(f"[info] summary columns: {list(wide.columns)}")

    labels = [f"ε={eps}" for eps in wide["dp_epsilon"].tolist()]
    y = np.arange(len(wide))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.4), sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.10, wspace=0.06, hspace=0.02)

    for ax, (metric_key, metric_name, title) in zip(axes, METRICS):
        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        if mean_col not in wide.columns:
            raise KeyError(
                f"Missing column {mean_col} in summary. Available columns: {list(wide.columns)}. "
                f"This likely means the metric '{metric_key}' was missing in the input JSON reports."
            )
        means = wide[mean_col].to_numpy(dtype=float)
        stds = wide.get(std_col, pd.Series([0.0] * len(wide))).fillna(0.0).to_numpy(dtype=float)

        # std only to the right (like the original two-panel plot)
        xerr = np.vstack([np.zeros_like(stds), stds])
        xmax = min(1.0, float(np.nanmax(means + stds)) + 0.12)

        ax.barh(
            y,
            means,
            xerr=xerr,
            color=COLOR_DP,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            capsize=4,
            alpha=0.84,
            zorder=3,
        )

        ax.set_title(title, pad=12)
        ax.set_xlabel("Mean score", labelpad=10)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, xmax)

        for yi, mean, std in zip(y, means, stds):
            xpos = mean + std + 0.02
            if xpos > xmax - 0.02:
                xpos = mean - 0.03
                ha = "right"
            else:
                ha = "left"
            ax.text(
                xpos,
                yi,
                f"{mean:.2f}",
                va="center",
                ha=ha,
                fontsize=FONT_SIZE - 2,
                color="#333333",
                zorder=4,
            )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("DP-query epsilon", labelpad=12)

    legend_handles = [
        Patch(facecolor=COLOR_DP, edgecolor=EDGE_COLOR, label="DP-query PREMS"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right", frameon=True)

    plt.savefig(OUT_IMG, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[OK] plot: {OUT_IMG}")


if __name__ == "__main__":
    main()
