#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export a small LaTeX snippet with example selected rules for a dataset/instance.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULESEL_DIR = Path(os.environ.get("RULESEL_DIR", str(PROJECT_ROOT / "Reports-eval" / "supporting-rule-selection"))).resolve()
DATASET = os.environ.get("DATASET", "spotify-r").strip()
ROW_ID_RAW = os.environ.get("ROW_ID", "").strip()
N_RULES = int(os.environ.get("N_RULES", "2"))

COVERAGE_PATH = RULESEL_DIR / "coverage_selected_supporting_rules.csv"
SIMPL_PATH = RULESEL_DIR / "selected_supporting_rules_simplified.csv"

USE_COVERAGE_ENV = os.environ.get("USE_COVERAGE", "").strip().lower()
if USE_COVERAGE_ENV in {"1", "true", "yes", "y"}:
    USE_COVERAGE = True
elif USE_COVERAGE_ENV in {"0", "false", "no", "n"}:
    USE_COVERAGE = False
else:
    USE_COVERAGE = COVERAGE_PATH.exists()


def main() -> None:
    if USE_COVERAGE:
        in_path = COVERAGE_PATH
        rank_col = "selected_rank_coverage"
    else:
        in_path = SIMPL_PATH if SIMPL_PATH.exists() else (RULESEL_DIR / "selected_supporting_rules.csv")
        rank_col = "selected_rank"

    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    df = pd.read_csv(in_path)
    df = df[df["dataset"] == DATASET].copy()
    if df.empty:
        raise ValueError(f"No rows for dataset={DATASET} in {in_path}")

    if ROW_ID_RAW:
        row_id = int(ROW_ID_RAW)
        df = df[df["row_id"] == row_id].copy()
        if df.empty:
            raise ValueError(f"No rows for dataset={DATASET}, row_id={row_id} in {in_path}")
    else:
        row_id = int(df["row_id"].iloc[0])
        df = df[df["row_id"] == row_id].copy()

    rule_col = "rendered_rule_simplified" if "rendered_rule_simplified" in df.columns else "rendered_rule"
    if rule_col not in df.columns:
        raise ValueError(f"Missing rule text column in {in_path}. Columns: {list(df.columns)}")

    if rank_col not in df.columns:
        # fallback: make a rank by score
        df = df.sort_values(["score", "leaf_purity", "leaf_support_train"], ascending=False).copy()
        df[rank_col] = range(1, len(df) + 1)
    else:
        df = df.sort_values(rank_col).copy()

    df = df.head(N_RULES).copy()

    # Build LaTeX snippet.
    lines = []
    lines.append(rf"\paragraph{{Example selected rules (dataset \texttt{{{DATASET}}}).}}")
    lines.append(r"We report two representative rules returned by the selection step (same explained instance).")
    lines.append("")

    for _, r in df.iterrows():
        rank = int(r[rank_col])
        rule_text = str(r[rule_col])
        lines.append(r"\begin{quote}")
        lines.append(r"\footnotesize")
        lines.append(rf"\textbf{{Rule {rank} ({rank_col}={rank}).}}\\")
        lines.append(rf"\texttt{{\detokenize{{{rule_text}}}}}")
        lines.append(r"\end{quote}")
        lines.append("")

    out_path = RULESEL_DIR / f"example_selected_rules_{DATASET}_row{row_id}.tex"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

