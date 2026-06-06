#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# CONFIG
IN_CSV = Path("../../plots/synthetic_quality_metrics_long.csv")

OUT_DIR = Path("../../plots/performance-plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMG = OUT_DIR / "synthetic_quality_two_panel.png"
OUT_CSV = OUT_DIR / "synthetic_quality_two_panel_summary.csv"

FONT_SIZE = 16

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 2,
})

METHOD_ORDER = [
    "entropy 25",
    "entropy 50",
    "margin",
    "logit",
    "kappa",
    "dpgan",
    "aim",
    "mst",
    "privbayes",
]

METHOD_LABELS = {
    "entropy 25": "Entropy 25",
    "entropy 50": "Entropy 50",
    "margin": "Margin",
    "logit": "Logit",
    "kappa": "Kappa",
    "dpgan": "DPGAN",
    "aim": "AIM",
    "mst": "MST",
    "privbayes": "PrivBayes",
}

PREMS_METHODS = {
    "entropy 25",
    "entropy 50",
    "margin",
    "logit",
    "kappa",
}

COLOR_PREMS = "#CC79A7"   # pastel pink
COLOR_COMP = "#7B8CCB"    # pastel blue-violet
EDGE_COLOR = "#4A4A4A"
GRID_COLOR = "#DDDDDD"

METRICS = ["XGB synth", "NN distance"]

# LOAD + AGGREGATE
df = pd.read_csv(IN_CSV)
df = df[df["metric"].isin(METRICS)].copy()

if df.empty:
    raise ValueError(f"No data found for metrics: {METRICS}")

agg = (
    df.groupby(["source", "method", "metric"])["value"]
      .agg(["mean", "std"])
      .reset_index()
)

wide_mean = agg.pivot_table(
    index=["source", "method"],
    columns="metric",
    values="mean"
).reset_index()

wide_std = agg.pivot_table(
    index=["source", "method"],
    columns="metric",
    values="std"
).reset_index()

wide = wide_mean.merge(
    wide_std,
    on=["source", "method"],
    suffixes=("_mean", "_std")
)

wide = wide[wide["method"].isin(METHOD_ORDER)].copy()
wide["method_order"] = wide["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
wide = wide.sort_values("method_order").reset_index(drop=True)

wide.to_csv(OUT_CSV, index=False)

labels = [METHOD_LABELS.get(m, m) for m in wide["method"]]
colors = [
    COLOR_PREMS if m in PREMS_METHODS else COLOR_COMP
    for m in wide["method"]
]

y = np.arange(len(wide))

# PLOT
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.8, 6.8),
    sharey=True,
    constrained_layout=True,
)

plot_specs = [
    ("XGB synth", "XGB synth. performance"),
    ("NN distance", "Nearest neigh. separation"),
]

for ax, (metric, title) in zip(axes, plot_specs):
    means = wide[f"{metric}_mean"].to_numpy(dtype=float)
    stds = wide[f"{metric}_std"].fillna(0).to_numpy(dtype=float)

    # std solo verso destra
    xerr = np.vstack([
        np.zeros_like(stds),
        stds,
    ])

    xmax = min(1.0, float(np.nanmax(means + stds)) + 0.12)

    ax.barh(
        y,
        means,
        xerr=xerr,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        capsize=4,
        alpha=0.84,
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xlabel("Mean score")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, mean, std in zip(y, means, stds):
        xpos = mean + std + 0.02

        # evita che il testo esca dal grafico
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
            fontsize=FONT_SIZE - 3,
            color="#333333",
            zorder=4,
        )

axes[0].set_yticks(y)
axes[0].set_yticklabels(labels)
axes[0].invert_yaxis()
axes[0].set_ylabel("Generation method")

legend_handles = [
    Patch(facecolor=COLOR_PREMS, edgecolor=EDGE_COLOR, label="PREMs"),
    Patch(facecolor=COLOR_COMP, edgecolor=EDGE_COLOR, label="Competitors"),
]

axes[1].legend(
    handles=legend_handles,
    loc="lower right",
    frameon=True,
)

plt.savefig(OUT_IMG, dpi=240, bbox_inches="tight")
plt.close()

print(f"[OK] summary: {OUT_CSV}")
print(f"[OK] plot: {OUT_IMG}")
