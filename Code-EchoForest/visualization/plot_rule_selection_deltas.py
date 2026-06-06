#!/usr/bin/env python3
"""
Plot the effect of rule-selection with before/after (dumbbell) markers.

This script is paper-oriented: it reads the cached before/after table and
produces a compact 3-panel dumbbell plot:
  (1) Rules per Instance (all vs selected)
  (2) Premises per Explanation (all vs selected)  [premises = literals]
  (3) Purity (all vs selected)

No CLI args on purpose (reproducibility on remote servers).
"""

from __future__ import annotations

import os
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
OUT_DIR = ROOT / "plots" / "explanations"
OUT_PNG = OUT_DIR / "rule_selection_deltas.png"
OUT_CSV = OUT_DIR / "rule_selection_deltas.csv"
IN_CSV = ROOT / "Reports-eval" / "supporting-rule-selection" / "rule_selection_before_after_table.csv"

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

# Match the general pastel + dark-gray style used in the privacy dotplot.
PINK = "#CC79A7"
PURPLE = "#5E548E"
TEXT = "#3F3F3F"
GRID = "#D9D9D9"
MARKER_SIZE = 110


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pretty_dataset(name: str) -> str:
    # Paper display names.
    mapping = {
        "adult": "Income",
        "activity": "Activity",
        "pol": "Pol",
        "spotify": "Spotify",
        "spotify-r": "Spotify R",
        "wave-binary": "Wave B",
        "wave-binary2": "Wave B2",
        "wave-multi": "Wave M",
        "landsat": "Landsat B",
        "landsat2": "Landsat B2",
        "landsat-multi": "Landsat M",
        "electricity": "Elec",
    }
    if name in mapping:
        return mapping[name]
    if not name:
        return name
    return name[0].upper() + name[1:]


def build_df() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {IN_CSV}. "
            "Run Code/explainability/build_rule_selection_table.py first."
        )

    df = pd.read_csv(IN_CSV)
    required = {
        "dataset",
        "rules_per_inst_all",
        "rules_per_inst_sel",
        "literals_per_inst_all",
        "literals_per_inst_sel",
        "purity_all",
        "purity_sel",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {IN_CSV}: {sorted(missing)}")

    df = df.rename(
        columns={
            "literals_per_inst_all": "literals_all",
            "literals_per_inst_sel": "literals_sel",
            "rules_per_inst_all": "rules_all",
            "rules_per_inst_sel": "rules_sel",
        }
    )
    df["dataset_display"] = df["dataset"].astype(str).map(pretty_dataset)
    df["delta_rules"] = df["rules_all"] - df["rules_sel"]
    df["delta_literals"] = df["literals_all"] - df["literals_sel"]
    df["pct_literals_reduction"] = 100.0 * (1.0 - (df["literals_sel"] / df["literals_all"]))
    df["delta_purity"] = df["purity_sel"] - df["purity_all"]
    return df


def plot(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)

    # Order by premises (all) so the heaviest cases are easy to spot.
    df = df.sort_values("literals_all", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig_h = 5.0
    fig_w = 15.2
    fig, (ax0, ax1, ax2) = plt.subplots(
        ncols=3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [0.95, 1.25, 0.85], "wspace": 0.34},
    )

    # Panel 0: rules per instance (all vs selected)
    r_all = df["rules_all"].to_numpy(dtype=float)
    r_sel = df["rules_sel"].to_numpy(dtype=float)
    valid_r = np.isfinite(r_all) & np.isfinite(r_sel)
    for ya, ra, rs, ok in zip(y, r_all, r_sel, valid_r):
        if ok:
            ax0.plot([rs, ra], [ya, ya], color=GRID, lw=2.2, zorder=1)
    ax0.scatter(r_all[np.isfinite(r_all)], y[np.isfinite(r_all)], s=MARKER_SIZE, color=PINK, edgecolor="white", linewidth=1.2, zorder=3, label="All rules")
    ax0.scatter(r_sel[np.isfinite(r_sel)], y[np.isfinite(r_sel)], s=MARKER_SIZE, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=4, label="Selected")
    ax0.set_yticks(y)
    ax0.set_yticklabels(df["dataset_display"].tolist(), color=TEXT)
    ax0.set_xlabel("Rules per Instance", color=TEXT)
    ax0.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.8, zorder=0)
    # Put a single horizontal legend below the whole figure (avoid covering data).
    # We create it later on the figure, so remove the per-axis legend.
    if ax0.get_legend() is not None:
        ax0.get_legend().remove()
    # Same issue as premises: selected is much smaller than all -> log scale helps readability.
    ax0.set_xscale("log")
    r_min = float(np.nanmin(np.where(np.isfinite(r_sel), r_sel, np.nan)))
    r_max = float(np.nanmax(np.where(np.isfinite(r_all), r_all, np.nan)))
    ax0.set_xlim(max(1e-2, r_min * 0.7), r_max * 1.25)

    # Panel 1: premises (= literals) per explanation (all vs selected)
    x_all = df["literals_all"].to_numpy(dtype=float)
    x_sel = df["literals_sel"].to_numpy(dtype=float)

    # Connectors
    for ya, xa, xs in zip(y, x_all, x_sel):
        ax1.plot([xs, xa], [ya, ya], color=GRID, lw=2.2, zorder=1)

    ax1.scatter(x_all, y, s=MARKER_SIZE, color=PINK, edgecolor="white", linewidth=1.2, zorder=3, label="All rules")
    ax1.scatter(x_sel, y, s=MARKER_SIZE, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=4, label="Selected")
    ax1.set_yticks(y)
    ax1.set_yticklabels([])  # share labels from leftmost panel
    ax1.set_xlabel("Premises per Explanation", color=TEXT)
    ax1.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.8, zorder=0)
    # The selected literals are often 1-2 orders of magnitude smaller than the full set.
    # Use a log scale so the "before/after" gaps are visually readable.
    ax1.set_xscale("log")
    xmin = max(1e-2, float(np.min(x_sel)) * 0.7)
    xmax = float(np.max(x_all)) * 1.15
    ax1.set_xlim(xmin, xmax)

    # Panel 2: purity (all vs selected)
    p_all = df["purity_all"].to_numpy(dtype=float)
    p_sel = df["purity_sel"].to_numpy(dtype=float)

    valid_p = np.isfinite(p_all) & np.isfinite(p_sel)
    for ya, pa, ps, ok in zip(y, p_all, p_sel, valid_p):
        if ok:
            ax2.plot([pa, ps], [ya, ya], color=GRID, lw=2.2, zorder=1)

    # Plot points even if one side is missing; if "all" is missing, show a subtle hollow marker.
    mask_all = np.isfinite(p_all)
    mask_sel = np.isfinite(p_sel)
    ax2.scatter(p_all[mask_all], y[mask_all], s=MARKER_SIZE, color=PINK, edgecolor="white", linewidth=1.2, zorder=3)
    ax2.scatter(p_sel[mask_sel], y[mask_sel], s=MARKER_SIZE, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=4)
    missing_all = (~mask_all) & mask_sel
    if np.any(missing_all):
        ax2.scatter(p_sel[missing_all], y[missing_all], s=MARKER_SIZE, facecolor="none", edgecolor=PINK, linewidth=1.6, zorder=5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])  # share labels from left panel
    ax2.set_xlabel("Purity", color=TEXT)
    ax2.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.8, zorder=0)

    # Add a bit of headroom so points at 1.0 are not clipped.
    ax2.set_xlim(0.65, 1.02)

    # Style cleanup
    for ax in (ax0, ax1, ax2):
        ax.tick_params(axis="x", colors=TEXT)
        ax.tick_params(axis="y", colors=TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(TEXT)
        ax.spines["bottom"].set_color(TEXT)

    # Use tight layout to keep it compact; bbox_inches on save handles legend.
    # One shared legend for all panels.
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=1.6,
        handletextpad=0.6,
    )

    # Leave room at the bottom for the shared legend.
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    df_out = df[
        [
            "dataset",
            "rules_all",
            "rules_sel",
            "delta_rules",
            "literals_all",
            "literals_sel",
            "delta_literals",
            "pct_literals_reduction",
            "purity_all",
            "purity_sel",
            "delta_purity",
        ]
    ].copy()
    df_out.to_csv(OUT_CSV, index=False)

    print(f"[OK] wrote {OUT_PNG}")
    print(f"[OK] wrote {OUT_CSV}")


def main() -> None:
    df = build_df()
    plot(df)


if __name__ == "__main__":
    main()
