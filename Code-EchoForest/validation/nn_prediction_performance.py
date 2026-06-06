
"""
Summarize prediction performance of the original neural-network black-boxes.

"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = [
    "income",
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

MODEL_DIR = Path("../Model-original")
OUT_DIR = Path("../Reports-eval") / "validation"
OUT_CSV = OUT_DIR / "nn_prediction_performance.csv"


def parse_report(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    acc_match = re.search(r"accuracy\s+([0-9]*\.[0-9]+)", text)
    f1_match = re.search(
        r"macro avg\s+[0-9]*\.[0-9]+\s+[0-9]*\.[0-9]+\s+([0-9]*\.[0-9]+)",
        text,
    )
    if acc_match is None or f1_match is None:
        raise ValueError(f"Could not parse accuracy/macro-F1 from {path}")

    return float(acc_match.group(1)), float(f1_match.group(1))


def resolve_dataset_dir(dataset: str) -> Path:
    if dataset == "income":
        return MODEL_DIR / "adult"
    return MODEL_DIR / dataset


def find_report(dataset: str, split: str) -> Path | None:
    ddir = resolve_dataset_dir(dataset)
    base_name = "adult" if dataset == "income" else dataset
    direct = ddir / f"nn_{base_name}_report_{split}.txt"
    if direct.exists():
        return direct

    matches = sorted(ddir.glob(f"nn_*_report_{split}.txt"))
    return matches[0] if matches else None


def collect_rows() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        row = {"dataset": dataset}
        for split in ("train", "test"):
            report = find_report(dataset, split)
            if report is None:
                row[f"{split}_accuracy"] = np.nan
                row[f"{split}_macro_f1"] = np.nan
                row[f"{split}_report_path"] = ""
                continue

            try:
                acc, macro_f1 = parse_report(report)
                row[f"{split}_accuracy"] = acc
                row[f"{split}_macro_f1"] = macro_f1
                row[f"{split}_report_path"] = str(report)
            except Exception:
                row[f"{split}_accuracy"] = np.nan
                row[f"{split}_macro_f1"] = np.nan
                row[f"{split}_report_path"] = str(report)
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(collect_rows())
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
