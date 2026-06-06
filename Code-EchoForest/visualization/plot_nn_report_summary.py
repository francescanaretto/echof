#!/usr/bin/env python3
"""
Create a compact radar-like summary plot of NN macro-F1 across datasets.

The script parses the precomputed text reports stored in Model-original, rather
than recomputing predictions from the saved models. This keeps the figure
aligned with the results already validated in the project.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_FONT_SIZE = 15
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

# Make the Code root importable when the script is executed directly.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.project_paths import PROJECT_ROOT


# CONFIG
DATASETS = [
    "adult",
    "activity",
    "pol",
    "spotify",
    "spotify-r",
    "wave-binary",
    "wave-binary2",
    "credit",
    "diamonds_binary",
    "htru2",
    "landsat",
    "landsat2",
    "magic",
    "heloc",
    "electricity",
    "california",
    "splice",
    "wave-multi",
    "landsat-multi",
    "pendigits",
]

MODEL_DIR = PROJECT_ROOT / "Model-original"
OUT_DIR = PROJECT_ROOT / "plots" / "nn_report_summary"
OUT_PNG = OUT_DIR / "nn_test_macro_f1_radar.png"
OUT_CSV = OUT_DIR / "nn_test_performance_summary.csv"

FIGSIZE = (6.2, 6.2)
DPI = 220

COLOR_F1 = "#F2B5D4"    # pastel pink
COLOR_FILL = "#F8DCEC"
GRID_COLOR = "#D9D9D9"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prettify_dataset_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def parse_report(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    acc_match = re.search(r"accuracy\s+([0-9]*\.[0-9]+)", text)
    f1_match = re.search(r"macro avg\s+[0-9]*\.[0-9]+\s+[0-9]*\.[0-9]+\s+([0-9]*\.[0-9]+)", text)

    if acc_match is None or f1_match is None:
        raise ValueError(f"Could not parse accuracy/macro-F1 from {path}")

    return float(acc_match.group(1)), float(f1_match.group(1))


def find_report(dataset: str) -> Path | None:
    direct = MODEL_DIR / dataset / f"nn_{dataset}_report_test.txt"
    if direct.exists():
        return direct

    matches = sorted((MODEL_DIR / dataset).glob("nn_*_report_test.txt"))
    if matches:
        return matches[0]
    return None


def collect_results() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        report_path = find_report(dataset)
        if report_path is None:
            rows.append(
                {
                    "dataset": dataset,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "report_path": "",
                    "error": "missing report",
                }
            )
            continue

        try:
            accuracy, macro_f1 = parse_report(report_path)
            rows.append(
                {
                    "dataset": dataset,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "report_path": str(report_path),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "report_path": str(report_path),
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def shorten_dataset_name(name: str) -> str:
    mapping = {
        "spotify-r": "Spotify-r",
        "wave-binary": "Wave-bin",
        "wave-binary2": "Wave-bin2",
        "landsat-multi": "Landsat-m",
        "wave-multi": "Wave-m",
        "diamonds_binary": "Diamonds",
    }
    if name in mapping:
        return mapping[name]
    pretty = name.replace("_", " ").replace("-", " ")
    return pretty[:1].upper() + pretty[1:]


def plot_summary(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)

    plot_df = df.copy()
    plot_df = plot_df[np.isfinite(plot_df["macro_f1"])].copy()
    plot_df = plot_df.sort_values("macro_f1", ascending=False)
    plot_df["Label"] = plot_df["dataset"].map(shorten_dataset_name)

    scores = plot_df["macro_f1"].to_numpy(dtype=float)
    labels = plot_df["Label"].tolist()
    n = len(scores)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    scores_closed = np.r_[scores, scores[0]]
    angles_closed = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={"projection": "polar"}, constrained_layout=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, scores_closed, color=COLOR_F1, linewidth=1.8)
    ax.fill(angles_closed, scores_closed, color=COLOR_FILL, alpha=0.75)
    ax.scatter(angles, scores, s=18, color=COLOR_F1, edgecolor="#8C5A76", linewidth=0.5, zorder=3)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=BASE_FONT_SIZE - 2)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_title("NN Macro-F1 Across Datasets", pad=18, fontsize=BASE_FONT_SIZE)

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(OUT_DIR)
    df = collect_results()
    df.to_csv(OUT_CSV, index=False)
    plot_summary(df)
    print(f"[OK] wrote {OUT_CSV}")
    print(f"[OK] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
