#!/usr/bin/env python3
"""Privacy-risk reduction dumbbell plot for selected datasets."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter

BASE_FONT_SIZE = 19
LEGEND_FONT_SIZE = 14
ROW_STEP = 0.78
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

THIS_FILE = Path(__file__).resolve()
CODE_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = CODE_ROOT.parent

PRIVACY_DIR = PROJECT_ROOT / "privacy"
OUT_DIR = PROJECT_ROOT / "plots" / "privacy_delta_dotplot"

TARGET_DATASETS = [
    "splice",
    "credit",
    "heloc",
    "electricity",
]
TARGET_GUIDING_BB = "nn"
TARGET_SHADOW_MODE = "direct_rf"
TARGET_SHADOW_DATA = "synth"

ATTACK_ORDER = ["loss", "conf", "ent", "marg"]
ATTACK_LABELS = {
    "loss": "Loss",
    "conf": "Confidence",
    "ent": "Entropy",
    "marg": "Margin",
}
ATTACK_COLORS = {
    "loss": "#F4C7AB",
    "conf": "#BFD7EA",
    "ent": "#D7EAD3",
    "marg": "#D9D2F0",
}
CONNECTOR_BASE_COLOR = "#C04A7A"
MARKER_SIZE = 78

ORIGINAL_COLOR = "#6E7F80"
PREMS_COLOR = "#D46A6A"

DATASET_LABELS = {
    "adult": "Adult",
    "california": "California",
    "credit": "Credit",
    "electricity": "Electricity",
    "heloc": "Heloc",
    "letters": "Letters",
    "splice": "Splice",
}

FILENAME_RE = re.compile(
    r"^privacy_report_"
    r"(?P<dataset>.+?)_"
    r"(?P<kind>.+?)_"
    r"(?P<guiding_bb>.+?)_"
    r"(?P<percentile>\d+)"
    r"_premsShadowMode-(?P<shadow_mode>.+?)"
    r"_premsShadowData-(?P<shadow_data>.+?)"
    r"\.txt$"
)

SECTION_RE_TEMPLATE = (
    r"--- Membership Inference \({section} attack\) ---\s+"
    r"NN originale:\s+AUC=(?P<nn>[0-9.]+)\s+\|\s+TPR@1%FPR=[0-9.]+\s+"
    r"PREMs RF:\s+AUC=(?P<prems>[0-9.]+)\s+\|\s+TPR@1%FPR=[0-9.]+"
)


def dataset_label(name: str) -> str:
    return DATASET_LABELS.get(name, name[:1].upper() + name[1:])


def parse_file_metadata(path: Path) -> dict | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    meta = match.groupdict()
    if meta["dataset"] not in TARGET_DATASETS:
        return None
    if meta["guiding_bb"] != TARGET_GUIDING_BB:
        return None
    if meta["shadow_mode"] != TARGET_SHADOW_MODE:
        return None
    if meta["shadow_data"] != TARGET_SHADOW_DATA:
        return None
    return meta


def parse_attack_auc(text: str, attack: str) -> tuple[float, float] | None:
    section_name = {
        "loss": "LOSS",
        "conf": "CONFIDENCE",
        "ent": "ENTROPY",
        "marg": "MARGIN",
    }[attack]
    match = re.search(SECTION_RE_TEMPLATE.format(section=section_name), text, flags=re.MULTILINE)
    if not match:
        return None
    return float(match.group("nn")), float(match.group("prems"))


def collect_rows() -> list[dict]:
    best_by_dataset: dict[str, dict] = {}

    for path in sorted(PRIVACY_DIR.glob("privacy_report_*.txt")):
        meta = parse_file_metadata(path)
        if meta is None:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        dataset = meta["dataset"]

        per_attack = {}
        for attack in ATTACK_ORDER:
            parsed = parse_attack_auc(text, attack)
            if parsed is None:
                continue
            nn_auc, prems_auc = parsed
            per_attack[attack] = {
                "nn_auc": nn_auc,
                "prems_auc": prems_auc,
                "delta_auc": nn_auc - prems_auc,
            }

        if not per_attack:
            continue

        strongest_attack, strongest_values = max(
            per_attack.items(),
            key=lambda item: item[1]["delta_auc"],
        )

        row = {
            "dataset": dataset,
            "kind": meta["kind"],
            "percentile": int(meta["percentile"]),
            "strongest_attack": strongest_attack,
            "nn_auc": strongest_values["nn_auc"],
            "prems_auc": strongest_values["prems_auc"],
            "delta_auc": strongest_values["delta_auc"],
            "source_file": path.name,
        }

        if dataset not in best_by_dataset or row["delta_auc"] > best_by_dataset[dataset]["delta_auc"]:
            best_by_dataset[dataset] = row

    rows = []
    for dataset in TARGET_DATASETS:
        if dataset in best_by_dataset:
            rows.append(best_by_dataset[dataset])
    return rows


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "kind",
        "percentile",
        "strongest_attack",
        "nn_auc",
        "prems_auc",
        "delta_auc",
        "source_file",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No privacy rows available for the configured filters.")

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    y_positions = [idx * ROW_STEP for idx in range(len(rows))]
    max_x = max(max(row["nn_auc"], row["prems_auc"]) for row in rows)
    min_x = min(min(row["nn_auc"], row["prems_auc"]) for row in rows)

    for idx, row in enumerate(rows):
        y_pos = y_positions[idx]
        attack = row["strongest_attack"]
        guide_color = ATTACK_COLORS.get(attack, "#D9D9D9")
        nn_auc = row["nn_auc"]
        prems_auc = row["prems_auc"]

        line_start = min(prems_auc, nn_auc)
        line_end = max(prems_auc, nn_auc)
        ax.hlines(y_pos, xmin=line_start, xmax=line_end, color=CONNECTOR_BASE_COLOR, linewidth=3.8, zorder=1)
        ax.scatter(nn_auc, y_pos, s=MARKER_SIZE, color=ORIGINAL_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(prems_auc, y_pos, s=MARKER_SIZE, color=PREMS_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(
            max(nn_auc, prems_auc) + 0.004,
            y_pos,
            f"Δ={row['delta_auc']:.2f}",
            va="center",
            ha="left",
            fontsize=BASE_FONT_SIZE - 1,
            color="#333333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([dataset_label(row["dataset"]) for row in rows], fontsize=BASE_FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Attack AUC", fontsize=BASE_FONT_SIZE)
    ax.xaxis.set_major_locator(FixedLocator([0.50, 0.55, 0.60]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=BASE_FONT_SIZE)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(max(0.49, min_x - 0.006), min(0.62, max_x + 0.03))

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BFBFBF")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORIGINAL_COLOR, markeredgecolor="white",
               markeredgewidth=0.8, markersize=10, label="Original"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PREMS_COLOR, markeredgecolor="white",
               markeredgewidth=0.8, markersize=10, label="EchoForest"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        ncol=1,
        handlelength=2.2,
        columnspacing=1.2,
        labelspacing=0.8,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    csv_out = OUT_DIR / "privacy_delta_dotplot_summary.csv"
    png_out = OUT_DIR / "privacy_delta_dotplot.png"
    write_summary_csv(rows, csv_out)
    plot_rows(rows, png_out)
    print(f"[OK] wrote {csv_out}")
    print(f"[OK] wrote {png_out}")


if __name__ == "__main__":
    main()
