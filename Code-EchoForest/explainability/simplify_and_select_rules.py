#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process rule explanations to improve readability without recomputing rules.

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "Reports-eval" / "supporting-rule-selection"
RULESEL_DIR = Path(os.environ.get("RULESEL_DIR", str(DEFAULT_DIR))).resolve()

COVERAGE_TARGET = float(os.environ.get("COVERAGE_TARGET", "0.90"))
K_MAX = int(os.environ.get("K_MAX", "5"))
MAX_FEATURE_JACCARD = float(os.environ.get("MAX_FEATURE_JACCARD", "0.60"))


@dataclass(frozen=True)
class Cond:
    feature_idx: int
    feature_name: str
    op: str
    thr: float


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def parse_conditions(conditions_json: str) -> list[Cond]:
    arr = json.loads(conditions_json)
    out: list[Cond] = []
    for d in arr:
        out.append(
            Cond(
                feature_idx=int(d["feature_idx"]),
                feature_name=str(d.get("feature_name", d["feature_idx"])),
                op=str(d["operator"]),
                thr=float(d["threshold"]),
            )
        )
    return out


def canonicalize_conditions(conds: list[Cond]) -> list[Cond]:
    """
    Merge duplicated constraints on the same feature.

    For each feature:
      - keep only the tightest upper bound (<= min)
      - keep only the tightest lower bound (> max)
      - if both exist, keep both as an interval.
    """
    by_feat: dict[int, dict[str, Any]] = {}
    for c in conds:
        state = by_feat.setdefault(
            c.feature_idx,
            {"name": c.feature_name, "ub": None, "lb": None},
        )
        if c.op == "<=":
            state["ub"] = c.thr if state["ub"] is None else min(state["ub"], c.thr)
        elif c.op == ">":
            state["lb"] = c.thr if state["lb"] is None else max(state["lb"], c.thr)
        else:
            # unexpected operator: keep as-is by encoding it as a tight UB proxy
            state["ub"] = c.thr if state["ub"] is None else min(state["ub"], c.thr)

    out: list[Cond] = []
    for feat_idx, state in by_feat.items():
        name = state["name"]
        lb = state["lb"]
        ub = state["ub"]
        if lb is not None:
            out.append(Cond(feat_idx, name, ">", float(lb)))
        if ub is not None:
            out.append(Cond(feat_idx, name, "<=", float(ub)))

    # stable order: by feature index, then '>' before '<='
    out.sort(key=lambda c: (c.feature_idx, 0 if c.op == ">" else 1))
    return out


def render_rule(conds: list[Cond]) -> str:
    parts = [f"{c.feature_name} {c.op} {c.thr:.6f}" for c in conds]
    return " AND ".join(parts) if parts else "[root]"


def approx_union_coverage(supports: list[float]) -> float:
    """
    Overlap-agnostic approximation: P(union) = 1 - prod(1 - s_i)
    """
    p_not = 1.0
    for s in supports:
        s = float(np.clip(s, 0.0, 1.0))
        p_not *= (1.0 - s)
    return 1.0 - p_not


def simplify_selected_rules(df_selected: pd.DataFrame) -> pd.DataFrame:
    df = df_selected.copy()
    simplified_json = []
    simplified_rendered = []
    simplified_len = []
    simplified_unique = []
    for _, row in df.iterrows():
        conds = parse_conditions(row["conditions_json"])
        canon = canonicalize_conditions(conds)
        simplified_json.append(
            json.dumps(
                [
                    {
                        "feature_idx": c.feature_idx,
                        "operator": c.op,
                        "threshold": c.thr,
                        "feature_name": c.feature_name,
                    }
                    for c in canon
                ]
            )
        )
        simplified_rendered.append(render_rule(canon))
        simplified_len.append(len(canon))
        simplified_unique.append(len({c.feature_idx for c in canon}))

    df["conditions_json_simplified"] = simplified_json
    df["rendered_rule_simplified"] = simplified_rendered
    df["rule_length_simplified"] = simplified_len
    df["unique_features_simplified"] = simplified_unique
    return df


def coverage_select(group: pd.DataFrame) -> pd.DataFrame:
    """
    Select a diverse set of rules until approx coverage >= target (or k_max reached).
    Uses the *existing* ranking in the CSV (score descending).
    """
    ranked = group.sort_values(["score", "leaf_purity", "leaf_support_train"], ascending=False)
    selected_rows = []
    selected_feat_sets: list[set[int]] = []
    supports: list[float] = []

    for _, r in ranked.iterrows():
        feat_set = set(json.loads(r["feature_indices"]))
        max_j = max((jaccard(feat_set, s) for s in selected_feat_sets), default=0.0)
        if max_j > MAX_FEATURE_JACCARD:
            continue

        selected_rows.append(r)
        selected_feat_sets.append(feat_set)
        supports.append(float(r["leaf_support_train"]))

        if len(selected_rows) >= K_MAX:
            break

        if approx_union_coverage(supports) >= COVERAGE_TARGET:
            break

    if not selected_rows:
        return group.head(0)

    out = pd.DataFrame(selected_rows).copy()
    out["coverage_target"] = float(COVERAGE_TARGET)
    out["coverage_k_max"] = int(K_MAX)
    out["coverage_approx"] = float(approx_union_coverage(supports))
    out["selected_rank_coverage"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    in_path = RULESEL_DIR / "selected_supporting_rules.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)
    required = {"conditions_json", "feature_indices", "leaf_support_train", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    out_simpl = RULESEL_DIR / "selected_supporting_rules_simplified.csv"
    out_cov = RULESEL_DIR / "coverage_selected_supporting_rules.csv"

    df_simpl = simplify_selected_rules(df)
    df_simpl.to_csv(out_simpl, index=False)
    print(f"[OK] wrote {out_simpl}")

    group_cols = [
        "dataset", "kind", "guiding_bb", "percentile",
        "privacy_mode", "query_epsilon", "query_noise_mech", "noise_on_labeling",
        "model_case_id", "row_id",
    ]
    cov_frames = []
    for _, g in df_simpl.groupby(group_cols, dropna=False):
        cov_frames.append(coverage_select(g))
    df_cov = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame()
    df_cov.to_csv(out_cov, index=False)
    print(f"[OK] wrote {out_cov}")


if __name__ == "__main__":
    main()

