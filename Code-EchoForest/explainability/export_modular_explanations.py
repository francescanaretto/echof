#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export modular explanation views from already-extracted rule CSVs.

"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULESEL_DIR = PROJECT_ROOT / "Reports-eval" / "supporting-rule-selection"
RULESEL_DIR = Path(os.environ.get("RULESEL_DIR", str(DEFAULT_RULESEL_DIR))).resolve()

# Output directory:
# - prefer the newer layout Reports-eval/modular if present
# - otherwise fall back to Reports-eval/supporting-rule-selection/modular
DEFAULT_MODULAR_DIR = PROJECT_ROOT / "Reports-eval" / "modular"
FALLBACK_MODULAR_DIR = RULESEL_DIR / "modular"
_mod_env = os.environ.get("MODULAR_DIR", "").strip()
if _mod_env:
    OUT_DIR = Path(_mod_env)
    OUT_DIR = (OUT_DIR if OUT_DIR.is_absolute() else (PROJECT_ROOT / OUT_DIR)).resolve()
else:
    OUT_DIR = (DEFAULT_MODULAR_DIR if DEFAULT_MODULAR_DIR.exists() else FALLBACK_MODULAR_DIR).resolve()

DATASETS = [
    d.strip()
    for d in os.environ.get(
        "DATASETS",
        "activity,adult,electricity,landsat,landsat2,landsat-multi,pol,spotify,spotify-r,wave-binary,wave-multi",
    ).split(",")
    if d.strip()
]

TARGET_K = str(os.environ.get("TARGET_K", "5")).strip()

FILTER_KIND = os.environ.get("FILTER_KIND", "").strip() or None
FILTER_GUIDING_BB = os.environ.get("FILTER_GUIDING_BB", "").strip() or None
FILTER_PERCENTILE = os.environ.get("FILTER_PERCENTILE", "").strip() or None
FILTER_PRIVACY_MODE = os.environ.get("FILTER_PRIVACY_MODE", "").strip() or None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def pass_filters(row: Dict[str, str]) -> bool:
    if FILTER_KIND is not None and row.get("kind") != FILTER_KIND:
        return False
    if FILTER_GUIDING_BB is not None and row.get("guiding_bb") != FILTER_GUIDING_BB:
        return False
    if FILTER_PERCENTILE is not None:
        # be tolerant: "25" vs "25.0"
        if row.get("percentile") is None:
            return False
        if str(row.get("percentile")).strip() not in {FILTER_PERCENTILE, f"{FILTER_PERCENTILE}.0"}:
            return False
    if FILTER_PRIVACY_MODE is not None and row.get("privacy_mode") != FILTER_PRIVACY_MODE:
        return False
    return True


def choose_input_csv() -> Tuple[Path, str]:
    """
    Choose the best available per-rule CSV as input for top-k exports.
    Preference:
      1) selected_supporting_rules_simplified.csv
      2) selected_supporting_rules.csv
    """
    p1 = RULESEL_DIR / "selected_supporting_rules_simplified.csv"
    if p1.exists():
        return p1, "simplified"
    p0 = RULESEL_DIR / "selected_supporting_rules.csv"
    if p0.exists():
        return p0, "raw"
    raise FileNotFoundError(f"Missing inputs in {RULESEL_DIR}: selected_supporting_rules*.csv")


def choose_coverage_csv() -> Optional[Path]:
    p = RULESEL_DIR / "coverage_selected_supporting_rules.csv"
    return p if p.exists() else None


def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


@dataclass
class FeatureAgg:
    feature_idx: int
    feature_name: str
    lb: Optional[float] = None  # max of '>' thresholds
    ub: Optional[float] = None  # min of '<=' thresholds
    n_constraints: int = 0
    n_rules_with_feature: int = 0


def update_feature_agg(agg: FeatureAgg, conds: List[Dict[str, Any]]) -> None:
    """
    Update agg with conditions from a single rule.
    We treat:
      - operator ">" as lower bound
      - operator "<=" as upper bound
    """
    seen_in_rule = False
    for c in conds:
        op = str(c.get("operator"))
        thr = c.get("threshold")
        if thr is None:
            continue
        try:
            thr_f = float(thr)
        except Exception:
            continue

        if op == ">":
            agg.lb = thr_f if agg.lb is None else max(agg.lb, thr_f)
            agg.n_constraints += 1
            seen_in_rule = True
        elif op == "<=":
            agg.ub = thr_f if agg.ub is None else min(agg.ub, thr_f)
            agg.n_constraints += 1
            seen_in_rule = True
        else:
            # ignore unexpected operators for the feature-view summary
            continue

    if seen_in_rule:
        agg.n_rules_with_feature += 1


def render_feature_interval(agg: FeatureAgg) -> str:
    if agg.lb is not None and agg.ub is not None:
        return f"({agg.lb:.6f}, {agg.ub:.6f}]"
    if agg.lb is not None:
        return f"({agg.lb:.6f}, +inf)"
    if agg.ub is not None:
        return f"(-inf, {agg.ub:.6f}]"
    return "(-inf, +inf)"


def export_top1_per_dataset(in_csv: Path, mode: str) -> Dict[str, Path]:
    """
    Writes top-1 rule per instance. Returns dict dataset->output path.
    """
    out_paths: Dict[str, Path] = {}
    for ds in DATASETS:
        ds_dir = OUT_DIR / ds
        ensure_dir(ds_dir)
        out_paths[ds] = ds_dir / "top1_rules.csv"

        with out_paths[ds].open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "privacy_mode",
                    "kind",
                    "guiding_bb",
                    "percentile",
                    "query_epsilon",
                    "query_noise_mech",
                    "noise_on_labeling",
                    "model_case_id",
                    "row_id",
                    "rf_pred",
                    "selected_rank",
                    "rule_length",
                    "unique_features",
                    "leaf_support_train",
                    "leaf_purity",
                    "score",
                    "rendered_rule",
                ]
            )

    rule_col = "rendered_rule_simplified" if mode == "simplified" else "rendered_rule"
    len_col = "rule_length_simplified" if mode == "simplified" else "rule_length"
    uniq_col = "unique_features_simplified" if mode == "simplified" else "unique_features"

    for row in iter_csv_rows(in_csv):
        ds = row.get("dataset", "")
        if ds not in out_paths:
            continue
        if not pass_filters(row):
            continue
        if row.get("target_k") != TARGET_K:
            continue
        if row.get("selected_rank") != "1":
            continue

        out = out_paths[ds]
        with out.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    ds,
                    row.get("privacy_mode"),
                    row.get("kind"),
                    row.get("guiding_bb"),
                    row.get("percentile"),
                    row.get("query_epsilon"),
                    row.get("query_noise_mech"),
                    row.get("noise_on_labeling"),
                    row.get("model_case_id"),
                    row.get("row_id"),
                    row.get("rf_pred"),
                    row.get("selected_rank"),
                    row.get(len_col),
                    row.get(uniq_col),
                    row.get("leaf_support_train"),
                    row.get("leaf_purity"),
                    row.get("score"),
                    row.get(rule_col),
                ]
            )

    return out_paths


def export_coverage_per_dataset(coverage_csv: Optional[Path], in_csv: Path, mode: str) -> Dict[str, Path]:
    """
    Writes coverage-driven rule selection per dataset.

    If coverage_csv is available, it is used directly (preferred).
    Otherwise, we fall back to exporting all selected rules (target_k),
    keeping their ranking, but we label it as a fallback.
    """
    out_paths: Dict[str, Path] = {}
    for ds in DATASETS:
        ds_dir = OUT_DIR / ds
        ensure_dir(ds_dir)
        out_paths[ds] = ds_dir / "coverage_rules.csv"
        with out_paths[ds].open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "privacy_mode",
                    "kind",
                    "guiding_bb",
                    "percentile",
                    "query_epsilon",
                    "query_noise_mech",
                    "noise_on_labeling",
                    "model_case_id",
                    "row_id",
                    "rf_pred",
                    "selected_rank_coverage",
                    "coverage_target",
                    "coverage_k_max",
                    "coverage_approx",
                    "rule_length",
                    "unique_features",
                    "leaf_support_train",
                    "leaf_purity",
                    "score",
                    "rendered_rule",
                    "source",
                ]
            )

    if coverage_csv is not None:
        rule_col = "rendered_rule_simplified" if "rendered_rule_simplified" in (next(iter_csv_rows(coverage_csv)).keys()) else "rendered_rule"
        len_col = "rule_length_simplified" if "rule_length_simplified" in (next(iter_csv_rows(coverage_csv)).keys()) else "rule_length"
        uniq_col = "unique_features_simplified" if "unique_features_simplified" in (next(iter_csv_rows(coverage_csv)).keys()) else "unique_features"

        for row in iter_csv_rows(coverage_csv):
            ds = row.get("dataset", "")
            if ds not in out_paths:
                continue
            if not pass_filters(row):
                continue
            out = out_paths[ds]
            with out.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        ds,
                        row.get("privacy_mode"),
                        row.get("kind"),
                        row.get("guiding_bb"),
                        row.get("percentile"),
                        row.get("query_epsilon"),
                        row.get("query_noise_mech"),
                        row.get("noise_on_labeling"),
                        row.get("model_case_id"),
                        row.get("row_id"),
                        row.get("rf_pred"),
                        row.get("selected_rank_coverage"),
                        row.get("coverage_target"),
                        row.get("coverage_k_max"),
                        row.get("coverage_approx"),
                        row.get(len_col),
                        row.get(uniq_col),
                        row.get("leaf_support_train"),
                        row.get("leaf_purity"),
                        row.get("score"),
                        row.get(rule_col),
                        "coverage_csv",
                    ]
                )
        return out_paths

    # Fallback: dump the selected rules for TARGET_K.
    rule_col = "rendered_rule_simplified" if mode == "simplified" else "rendered_rule"
    len_col = "rule_length_simplified" if mode == "simplified" else "rule_length"
    uniq_col = "unique_features_simplified" if mode == "simplified" else "unique_features"
    for row in iter_csv_rows(in_csv):
        ds = row.get("dataset", "")
        if ds not in out_paths:
            continue
        if not pass_filters(row):
            continue
        if row.get("target_k") != TARGET_K:
            continue
        out = out_paths[ds]
        with out.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    ds,
                    row.get("privacy_mode"),
                    row.get("kind"),
                    row.get("guiding_bb"),
                    row.get("percentile"),
                    row.get("query_epsilon"),
                    row.get("query_noise_mech"),
                    row.get("noise_on_labeling"),
                    row.get("model_case_id"),
                    row.get("row_id"),
                    row.get("rf_pred"),
                    row.get("selected_rank"),
                    "",  # coverage_target
                    "",  # coverage_k_max
                    "",  # coverage_approx
                    row.get(len_col),
                    row.get(uniq_col),
                    row.get("leaf_support_train"),
                    row.get("leaf_purity"),
                    row.get("score"),
                    row.get(rule_col),
                    "fallback_topk",
                ]
            )
    return out_paths


def build_feature_view_for_dataset(ds: str, rules_csv: Path, out_path: Path, mode: str) -> None:
    """
    Build a per-feature view from a per-dataset rules CSV (typically coverage_rules.csv).
    """
    # group by (model_case_id, row_id)
    by_inst: Dict[Tuple[str, str], Dict[int, FeatureAgg]] = {}
    inst_meta: Dict[Tuple[str, str], Dict[str, str]] = {}

    # Determine which JSON field to parse for conditions.
    # If the dataset rules CSV includes simplified conditions, prefer them.
    # (When the input is coverage_selected_supporting_rules.csv, it often does.)
    cond_json_key = None
    rule_str_key = "rendered_rule"

    # peek header
    with rules_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    # The modular export always includes rendered_rule; simplified JSON might not exist in the reduced schema.
    # We therefore reconstruct feature-view from the *original* rule JSON if present in the source artifacts.

    # Try to find the best source CSV to parse conditions from.
    # Prefer the simplified table if it exists, else raw selected rules.
    src_csv, src_mode = choose_input_csv()
    if src_mode == "simplified":
        cond_json_key = "conditions_json_simplified"
    else:
        cond_json_key = "conditions_json"

    # Build a lookup from (model_case_id,row_id,rendered_rule) -> conditions_json
    # We only need this for the selected rules we exported, which are small.
    needed_keys: set[Tuple[str, str, str]] = set()
    for row in iter_csv_rows(rules_csv):
        needed_keys.add((row.get("model_case_id", ""), row.get("row_id", ""), row.get("rendered_rule", "")))

    cond_lookup: Dict[Tuple[str, str, str], Any] = {}
    for row in iter_csv_rows(src_csv):
        if row.get("dataset") != ds:
            continue
        if not pass_filters(row):
            continue
        if row.get("target_k") != TARGET_K:
            continue
        rr = row.get("rendered_rule_simplified") if src_mode == "simplified" else row.get("rendered_rule")
        key = (row.get("model_case_id", ""), row.get("row_id", ""), rr or "")
        if key not in needed_keys:
            continue
        conds = safe_json_loads(row.get(cond_json_key, "") or "")
        if isinstance(conds, list):
            cond_lookup[key] = conds

    for row in iter_csv_rows(rules_csv):
        inst = (row.get("model_case_id", ""), row.get("row_id", ""))
        inst_meta.setdefault(
            inst,
            {
                "dataset": ds,
                "privacy_mode": row.get("privacy_mode", ""),
                "kind": row.get("kind", ""),
                "guiding_bb": row.get("guiding_bb", ""),
                "percentile": row.get("percentile", ""),
                "query_epsilon": row.get("query_epsilon", ""),
                "query_noise_mech": row.get("query_noise_mech", ""),
                "noise_on_labeling": row.get("noise_on_labeling", ""),
                "rf_pred": row.get("rf_pred", ""),
            },
        )

        rr = row.get("rendered_rule", "") or ""
        conds = cond_lookup.get((inst[0], inst[1], rr))
        if not isinstance(conds, list):
            continue

        feats = by_inst.setdefault(inst, {})
        # within a rule, we count features presence once through update_feature_agg
        per_rule_by_feat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for c in conds:
            try:
                fid = int(c.get("feature_idx"))
            except Exception:
                continue
            per_rule_by_feat[fid].append(c)

        for fid, cond_list in per_rule_by_feat.items():
            name = str(cond_list[0].get("feature_name", fid))
            agg = feats.get(fid)
            if agg is None:
                agg = FeatureAgg(feature_idx=fid, feature_name=name)
                feats[fid] = agg
            update_feature_agg(agg, cond_list)

    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "privacy_mode",
                "kind",
                "guiding_bb",
                "percentile",
                "query_epsilon",
                "query_noise_mech",
                "noise_on_labeling",
                "model_case_id",
                "row_id",
                "rf_pred",
                "feature_idx",
                "feature_name",
                "interval",
                "lb",
                "ub",
                "n_rules_with_feature",
                "n_constraints",
            ]
        )

        for (model_case_id, row_id), feats in sorted(by_inst.items(), key=lambda x: (x[0][0], int(x[0][1]) if str(x[0][1]).isdigit() else x[0][1])):
            meta = inst_meta.get((model_case_id, row_id), {})
            # sort features by frequency then index for stability
            for fid, agg in sorted(
                feats.items(),
                key=lambda kv: (-kv[1].n_rules_with_feature, kv[1].feature_idx),
            ):
                w.writerow(
                    [
                        ds,
                        meta.get("privacy_mode", ""),
                        meta.get("kind", ""),
                        meta.get("guiding_bb", ""),
                        meta.get("percentile", ""),
                        meta.get("query_epsilon", ""),
                        meta.get("query_noise_mech", ""),
                        meta.get("noise_on_labeling", ""),
                        model_case_id,
                        row_id,
                        meta.get("rf_pred", ""),
                        agg.feature_idx,
                        agg.feature_name,
                        render_feature_interval(agg),
                        "" if agg.lb is None else f"{agg.lb:.6f}",
                        "" if agg.ub is None else f"{agg.ub:.6f}",
                        agg.n_rules_with_feature,
                        agg.n_constraints,
                    ]
                )


def main() -> None:
    ensure_dir(OUT_DIR)

    in_csv, mode = choose_input_csv()
    cov_csv = choose_coverage_csv()

    print(f"[info] RULESEL_DIR={RULESEL_DIR}")
    print(f"[info] input_csv={in_csv} (mode={mode})")
    print(f"[info] coverage_csv={cov_csv if cov_csv is not None else '(missing -> fallback)'}")
    print(f"[info] DATASETS={DATASETS}")
    print(f"[info] TARGET_K={TARGET_K}")
    print(
        f"[info] filters: kind={FILTER_KIND} guiding_bb={FILTER_GUIDING_BB} "
        f"percentile={FILTER_PERCENTILE} privacy_mode={FILTER_PRIVACY_MODE}"
    )

    export_top1_per_dataset(in_csv, mode)
    cov_paths = export_coverage_per_dataset(cov_csv, in_csv, mode)

    # Feature view is derived from the per-dataset coverage rules (or fallback file).
    for ds in DATASETS:
        rules_path = cov_paths[ds]
        out_path = (OUT_DIR / ds / "feature_view.csv").resolve()
        build_feature_view_for_dataset(ds, rules_path, out_path, mode)
        print(f"[OK] wrote {out_path}")

    print(f"[done] outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
