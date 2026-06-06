#!/usr/bin/env python3
"""
Aggregate WISE distribution logs and create publication-friendly plots.

Input files:
  Data-synthetic/wise/**/wise_dist_summary_synthetic_19_checks_*.csv

Default behavior:
  - considers only standard (non-DP) synthetic_19_checks runs
  - filters to guiding_bb = nn
  - filters to kind = logit
  - filters to percentile = 25

Outputs:
  Reports-eval/distribution-contributions/
    - distribution_contribution_summary.csv
    - distribution_contributions_<tag>.png
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE_FONT_SIZE = 16
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE - 1,
    }
)

ROOT = Path(__file__).resolve().parents[2]
WISE_DIR = ROOT / "Data-synthetic" / "wise"
OUT_DIR = ROOT / "Reports-eval" / "distribution-contributions"

# Default paper-oriented filter.
FILTER_KIND = "logit"       # None to keep all
FILTER_GUIDING_BB = "nn"    # None to keep all
FILTER_PERCENTILE = "25"    # None to keep all


DIST_NAMES = {
    0: "normal",
    1: "lognormal",
    2: "gamma",
    3: "beta",
    4: "student_t",
    5: "exponential",
    6: "poisson_centered",
    7: "gaussian_mixture_2",
    8: "bernoulli_pm1",
    9: "gaussian_mixture_k",
    10: "pareto_signed",
    11: "weibull",
    12: "laplace",
    13: "cauchy_clipped",
    14: "triangular",
    15: "logistic",
    16: "multivariate_normal",
    17: "multivariate_mixture",
    18: "correlated_binary_pm1",
}


NAME_RE = re.compile(
    r"wise_dist_summary_synthetic_19_checks_(?P<dataset>.+?)_"
    r"(?P<kind>entropy|margin|logit|kappa)_"
    r"(?P<guiding_bb>nn|rf)_"
    r"(?P<percentile>25|50)\.csv$"
)


def read_dist_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "dist_id": int(row["dist_id"]),
                    "success": float(row["success"]),
                    "trials": float(row["trials"]),
                    "success_per_trial": float(row["success_per_trial"]),
                    "weight_final": float(row["weight_final"]),
                }
            )
    return rows


def find_matching_files() -> list[tuple[dict, Path]]:
    matches = []
    for path in WISE_DIR.rglob("wise_dist_summary_synthetic_19_checks_*.csv"):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        meta = m.groupdict()
        if FILTER_KIND and meta["kind"] != FILTER_KIND:
            continue
        if FILTER_GUIDING_BB and meta["guiding_bb"] != FILTER_GUIDING_BB:
            continue
        if FILTER_PERCENTILE and meta["percentile"] != FILTER_PERCENTILE:
            continue
        matches.append((meta, path))
    return sorted(matches, key=lambda x: (x[0]["dataset"], x[1].name))


def make_tag() -> str:
    parts = []
    if FILTER_KIND:
        parts.append(FILTER_KIND)
    if FILTER_GUIDING_BB:
        parts.append(FILTER_GUIDING_BB)
    if FILTER_PERCENTILE:
        parts.append(FILTER_PERCENTILE)
    return "_".join(parts) if parts else "all"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pretty_name(name: str) -> str:
    return name.replace("_", " ").title()


def main() -> None:
    files = find_matching_files()
    if not files:
        raise FileNotFoundError("No matching WISE distribution summaries found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = make_tag()

    stats = {
        dist_id: {
            "dist_name": DIST_NAMES[dist_id],
            "n_files": 0,
            "sum_weight_final": 0.0,
            "sum_success_per_trial": 0.0,
            "sum_success_share": 0.0,
            "success_shares": [],
        }
        for dist_id in DIST_NAMES
    }

    n_files_total = 0

    for meta, path in files:
        rows = read_dist_summary(path)
        n_files_total += 1
        total_success = sum(r["success"] for r in rows) or 1.0
        for r in rows:
            dist_id = r["dist_id"]
            stats[dist_id]["n_files"] += 1
            stats[dist_id]["sum_weight_final"] += r["weight_final"]
            stats[dist_id]["sum_success_per_trial"] += r["success_per_trial"]
            share = r["success"] / total_success
            stats[dist_id]["sum_success_share"] += share
            stats[dist_id]["success_shares"].append(share)

    summary_rows = []
    for dist_id, s in stats.items():
        n = max(s["n_files"], 1)
        shares = s["success_shares"]
        if shares:
            avg_share = sum(shares) / len(shares)
            var_share = sum((x - avg_share) ** 2 for x in shares) / len(shares)
            std_share = var_share ** 0.5
        else:
            avg_share = 0.0
            std_share = 0.0
        summary_rows.append(
            {
                "dist_id": dist_id,
                "dist_name": s["dist_name"],
                "n_files": s["n_files"],
                "avg_weight_final": f"{s['sum_weight_final'] / n:.8f}",
                "avg_success_per_trial": f"{s['sum_success_per_trial'] / n:.8f}",
                "avg_success_share": f"{avg_share:.8f}",
                "std_success_share": f"{std_share:.8f}",
            }
        )

    summary_rows.sort(key=lambda r: float(r["avg_weight_final"]), reverse=True)

    write_csv(
        OUT_DIR / "distribution_contribution_summary.csv",
        summary_rows,
        [
            "dist_id",
            "dist_name",
            "n_files",
            "avg_weight_final",
            "avg_success_per_trial",
            "avg_success_share",
            "std_success_share",
        ],
    )

    try:
        # Sort by accepted share, which is the easiest notion of contribution to explain.
        ordered = sorted(summary_rows, key=lambda r: float(r["avg_success_share"]), reverse=True)
        names = [pretty_name(r["dist_name"]) for r in ordered]
        avg_shares = [float(r["avg_success_share"]) for r in ordered]
        std_shares = [float(r["std_success_share"]) for r in ordered]

        fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
        mean_color = "#B8D8D8"
        mean_edge = "#5F8F8F"
        std_color = "#E5989B"

        y = list(range(len(names)))

        ax.barh(
            y,
            avg_shares,
            color=mean_color,
            edgecolor=mean_edge,
            linewidth=1.0,
            zorder=2,
        )
        ax.errorbar(
            avg_shares,
            y,
            xerr=[[0.0] * len(std_shares), std_shares],
            fmt="none",
            ecolor=std_color,
            elinewidth=2.0,
            capsize=4,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=BASE_FONT_SIZE)
        ax.invert_yaxis()
        ax.set_xlabel("Average of Accepted Records per Distribution\nand Std (%)", labelpad=10)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=BASE_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE)
        out_png = OUT_DIR / f"distribution_contributions_{tag}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot to: {out_png}")
    except ModuleNotFoundError:
        print("matplotlib not available; CSV summaries were still generated.")

    print(f"Wrote summaries to: {OUT_DIR}")


if __name__ == "__main__":
    main()
