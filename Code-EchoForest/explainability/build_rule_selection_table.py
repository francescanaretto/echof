#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the "before/after" rule-selection table from cached CSVs.

"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULESEL_DIR = Path(os.environ.get("RULESEL_DIR", str(PROJECT_ROOT / "Reports-eval" / "supporting-rule-selection"))).resolve()
TARGET_K = int(os.environ.get("TARGET_K", "5"))
SIMPLIFIED = os.environ.get("SIMPLIFIED", "0").strip() in {"1", "true", "True", "yes", "Y"}
COVERAGE = os.environ.get("COVERAGE", "0").strip() in {"1", "true", "True", "yes", "Y"}
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "200000"))

# Optional strict filters to replicate "one setting per dataset" tables.
FILTER_KIND = os.environ.get("FILTER_KIND", "").strip() or None
FILTER_GUIDING_BB = os.environ.get("FILTER_GUIDING_BB", "").strip() or None
FILTER_PERCENTILE = os.environ.get("FILTER_PERCENTILE", "").strip() or None
FILTER_PRIVACY_MODE = os.environ.get("FILTER_PRIVACY_MODE", "").strip() or None


def _norm_percentile(v) -> str:
    """
    Percentile values are sometimes serialized as 25 or 25.0 depending on the writer.
    Normalize to an integer-like string (e.g., "25") for robust filtering.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    except Exception:
        s = str(v).strip()
        if s.endswith(".0"):
            return s[:-2]
        return s


def _mean_purity_stream(path: Path, dataset_col: str = "dataset") -> pd.Series:
    """
    Stream-compute mean leaf_purity per dataset from a potentially huge CSV.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    usecols = [dataset_col, "leaf_purity", "kind", "guiding_bb", "percentile", "privacy_mode"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS):
        chunk[dataset_col] = chunk[dataset_col].astype(str)
        if FILTER_KIND is not None:
            chunk = chunk[chunk["kind"].astype(str) == FILTER_KIND]
        if FILTER_GUIDING_BB is not None:
            chunk = chunk[chunk["guiding_bb"].astype(str) == FILTER_GUIDING_BB]
        if FILTER_PERCENTILE is not None:
            want = _norm_percentile(FILTER_PERCENTILE)
            chunk = chunk[chunk["percentile"].map(_norm_percentile) == want]
        if FILTER_PRIVACY_MODE is not None:
            chunk = chunk[chunk["privacy_mode"].astype(str) == FILTER_PRIVACY_MODE]
        if chunk.empty:
            continue
        grp = chunk.groupby(dataset_col)["leaf_purity"].agg(["sum", "count"])
        for ds, row in grp.iterrows():
            sums[ds] = sums.get(ds, 0.0) + float(row["sum"])
            counts[ds] = counts.get(ds, 0) + int(row["count"])
    out = {ds: (sums[ds] / counts[ds]) for ds in sums if counts.get(ds, 0) > 0}
    return pd.Series(out, name="purity_mean")


def _selected_metrics_from_rules(path: Path, k: int) -> pd.DataFrame:
    """
    Compute per-dataset selected metrics from selected rules CSV.

    Returns columns:
      dataset, rules_per_inst_sel, literals_per_inst_sel, purity_sel
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    # pick columns depending on simplified mode
    length_col = "rule_length_simplified" if SIMPLIFIED and "rule_length_simplified" in pd.read_csv(path, nrows=1).columns else "rule_length"
    purity_col = "leaf_purity"

    usecols = ["dataset", "row_id", "target_k", length_col, purity_col, "kind", "guiding_bb", "percentile", "privacy_mode"]
    df = pd.read_csv(path, usecols=usecols)
    if FILTER_KIND is not None:
        df = df[df["kind"].astype(str) == FILTER_KIND]
    if FILTER_GUIDING_BB is not None:
        df = df[df["guiding_bb"].astype(str) == FILTER_GUIDING_BB]
    if FILTER_PERCENTILE is not None:
        want = _norm_percentile(FILTER_PERCENTILE)
        df = df[df["percentile"].map(_norm_percentile) == want]
    if FILTER_PRIVACY_MODE is not None:
        df = df[df["privacy_mode"].astype(str) == FILTER_PRIVACY_MODE]
    df = df[df["target_k"] == k].copy()

    per_inst = (
        df.groupby(["dataset", "row_id"], as_index=False)
          .agg(
              rules=("row_id", "size"),
              literals=(length_col, "sum"),
              purity=(purity_col, "mean"),
          )
    )
    per_ds = (
        per_inst.groupby("dataset", as_index=False)
               .agg(
                   rules_per_inst_sel=("rules", "mean"),
                   literals_per_inst_sel=("literals", "mean"),
                   purity_sel=("purity", "mean"),
               )
    )
    return per_ds


def _selected_metrics_from_coverage(path: Path) -> pd.DataFrame:
    """
    Compute per-dataset selected metrics from coverage-selected CSV.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    length_col = "rule_length_simplified" if SIMPLIFIED and "rule_length_simplified" in pd.read_csv(path, nrows=1).columns else "rule_length"
    usecols = ["dataset", "row_id", length_col, "leaf_purity", "kind", "guiding_bb", "percentile", "privacy_mode"]
    df = pd.read_csv(path, usecols=usecols)
    if FILTER_KIND is not None:
        df = df[df["kind"].astype(str) == FILTER_KIND]
    if FILTER_GUIDING_BB is not None:
        df = df[df["guiding_bb"].astype(str) == FILTER_GUIDING_BB]
    if FILTER_PERCENTILE is not None:
        want = _norm_percentile(FILTER_PERCENTILE)
        df = df[df["percentile"].map(_norm_percentile) == want]
    if FILTER_PRIVACY_MODE is not None:
        df = df[df["privacy_mode"].astype(str) == FILTER_PRIVACY_MODE]

    per_inst = (
        df.groupby(["dataset", "row_id"], as_index=False)
          .agg(
              rules=("row_id", "size"),
              literals=(length_col, "sum"),
              purity=("leaf_purity", "mean"),
          )
    )
    per_ds = (
        per_inst.groupby("dataset", as_index=False)
               .agg(
                   rules_per_inst_sel=("rules", "mean"),
                   literals_per_inst_sel=("literals", "mean"),
                   purity_sel=("purity", "mean"),
               )
    )
    return per_ds


def main() -> None:
    instance_path = RULESEL_DIR / "instance_selection_summary.csv"
    all_rules_path = RULESEL_DIR / "all_supporting_rules.csv"
    selected_rules_path = RULESEL_DIR / ("coverage_selected_supporting_rules.csv" if COVERAGE else "selected_supporting_rules.csv")
    selected_simpl_path = RULESEL_DIR / "selected_supporting_rules_simplified.csv"

    if not instance_path.exists():
        raise FileNotFoundError(str(instance_path))
    if not all_rules_path.exists():
        raise FileNotFoundError(str(all_rules_path))

    # 1) Rules/inst and Literals/inst from instance-level summaries (cheap)
    inst = pd.read_csv(instance_path)
    if FILTER_KIND is not None:
        inst = inst[inst["kind"].astype(str) == FILTER_KIND]
    if FILTER_GUIDING_BB is not None:
        inst = inst[inst["guiding_bb"].astype(str) == FILTER_GUIDING_BB]
    if FILTER_PERCENTILE is not None:
        want = _norm_percentile(FILTER_PERCENTILE)
        inst = inst[inst["percentile"].map(_norm_percentile) == want]
    if FILTER_PRIVACY_MODE is not None:
        inst = inst[inst["privacy_mode"].astype(str) == FILTER_PRIVACY_MODE]
    inst = inst[inst["target_k"] == TARGET_K].copy()

    base = (
        inst.groupby("dataset", as_index=False)
            .agg(
                rules_per_inst_all=("n_supporting_rules_full", "mean"),
                rules_per_inst_sel=("n_supporting_rules_selected", "mean"),
                literals_per_inst_all=("total_literals_full", "mean"),
                literals_per_inst_sel=("total_literals_selected", "mean"),
            )
    )

    # 2) Purity (all) from streaming big all_supporting_rules.csv
    purity_all = _mean_purity_stream(all_rules_path).rename("purity_all")
    base = base.merge(purity_all.reset_index().rename(columns={"index": "dataset"}), on="dataset", how="left")

    # 3) Purity (sel) and optionally simplified literal counts from selected rules
    if COVERAGE:
        sel_metrics = _selected_metrics_from_coverage(selected_rules_path)
    else:
        # Prefer simplified CSV if asked and present; otherwise fall back to selected_supporting_rules.csv
        if SIMPLIFIED and selected_simpl_path.exists():
            sel_metrics = _selected_metrics_from_rules(selected_simpl_path, TARGET_K)
        else:
            sel_metrics = _selected_metrics_from_rules(selected_rules_path, TARGET_K)

    base = base.drop(columns=["rules_per_inst_sel", "literals_per_inst_sel"], errors="ignore").merge(sel_metrics, on="dataset", how="left")

    # Formatting
    out = base.copy()
    # Keep stable dataset ordering
    out = out.sort_values("dataset").reset_index(drop=True)

    out_csv = RULESEL_DIR / "rule_selection_before_after_table.csv"
    out.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # LaTeX body (no table wrapper)
    def f2(x): return f"{float(x):.2f}"
    def f3(x): return f"{float(x):.3f}"

    lines = []
    for _, r in out.iterrows():
        lines.append(
            " & ".join(
                [
                    str(r["dataset"]),
                    f2(r["rules_per_inst_all"]),
                    f2(r["rules_per_inst_sel"]),
                    f2(r["literals_per_inst_all"]),
                    f2(r["literals_per_inst_sel"]),
                    f3(r["purity_all"]),
                    f3(r["purity_sel"]),
                ]
            )
            + r" \\"
        )
    out_tex = RULESEL_DIR / "rule_selection_before_after_table.tex"
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_tex}")


if __name__ == "__main__":
    main()
