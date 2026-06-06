#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract all supporting rules from PREMS Random Forest surrogates and select a
compact top-k diverse subset for each explained instance.


"""

from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Make `Code/` importable regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shared.project_paths import DATA_ORIGINAL_DIR, MODEL_SYNTHETIC_WISE_DIR, REPORTS_EVAL_DIR


DATASETS = [
    "adult",
    "activity",
    "landsat",
    "landsat2",
    "landsat-multi",
    "pol",
    "spotify",
    "spotify-r",
]

KIND = "logit"
GUIDING_BB = "nn"
PERCENTILE = 25

MODEL_SCOPE = "both"
# "standard" -> only standard models
# "dpquery"  -> only rf_dpquery_<dataset>_..._wise.sav
# "dp"       -> only rf_dp_<epsilon>_<mech>_<dataset>_... .sav
# "both"     -> standard + dpquery + dp

DIR_ORIGINAL = DATA_ORIGINAL_DIR
DIR_MODELS = MODEL_SYNTHETIC_WISE_DIR
DIR_REPORTS = REPORTS_EVAL_DIR / "supporting-rule-selection"

EVAL_SPLIT = "test"
MAX_EVAL_ROWS = 5000
RANDOM_STATE = 42

K_VALUES = [3, 5]
MAX_FEATURE_JACCARD = 0.60
FALLBACK_FILL = True
SAVE_RENDERED_RULES = True


@dataclass
class RuleCondition:
    feature_idx: int
    operator: str
    threshold: float


@dataclass
class RuleRecord:
    row_id: int
    rf_pred: int
    tree_idx: int
    leaf_idx: int
    rule_length: int
    unique_features: int
    leaf_support_train: float
    leaf_purity: float
    score: float
    feature_set: set[int]
    conditions: list[RuleCondition]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_reference_data(dataset: str, split: str = EVAL_SPLIT) -> tuple[np.ndarray, list[str]]:
    base = DIR_ORIGINAL / dataset
    train_df = pd.read_csv(base / f"train_set_{dataset}.csv")
    test_df = pd.read_csv(base / f"test_set_{dataset}.csv")

    train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.startswith("Unnamed")]

    feature_names = train_df.columns.tolist()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df.values.astype(np.float32))
    x_test = scaler.transform(test_df.values.astype(np.float32))

    if split == "train":
        x_ref = x_train
    elif split == "all":
        x_ref = np.vstack([x_train, x_test])
    else:
        x_ref = x_test

    if len(x_ref) > MAX_EVAL_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        chosen = np.sort(rng.choice(len(x_ref), size=MAX_EVAL_ROWS, replace=False))
        x_ref = x_ref[chosen]

    return x_ref.astype(np.float32), feature_names


def parse_model_filename(model_path: Path) -> Optional[dict]:
    name = model_path.name

    legacy_default = re.fullmatch(
        r"rf_(?P<dataset>.+)_wise\.sav",
        name,
    )
    if legacy_default:
        meta = legacy_default.groupdict()
        if meta["dataset"] in DATASETS:
            return {
                "privacy_mode": "standard",
                "dataset": meta["dataset"],
                "kind": "unknown",
                "guiding_bb": "unknown",
                "percentile": None,
                "query_epsilon": None,
                "query_noise_mech": None,
                "noise_on_labeling": None,
                "model_path": str(model_path),
                "model_name": name,
                "model_case_id": name.replace(".sav", ""),
            }

    standard = re.fullmatch(
        r"rf_(?P<dataset>.+)_(?P<kind>entropy|margin|kappa|logit)_(?P<bb>rf|nn)_(?P<percentile>\d+)_wise\.sav",
        name,
    )
    if standard:
        meta = standard.groupdict()
        return {
            "privacy_mode": "standard",
            "dataset": meta["dataset"],
            "kind": meta["kind"],
            "guiding_bb": meta["bb"],
            "percentile": int(meta["percentile"]),
            "query_epsilon": None,
            "query_noise_mech": None,
            "noise_on_labeling": None,
            "model_path": str(model_path),
            "model_name": name,
            "model_case_id": name.replace(".sav", ""),
        }

    dpquery = re.fullmatch(
        r"rf_dpquery_(?P<dataset>.+)_(?P<kind>entropy|margin|kappa|logit)_(?P<bb>rf|nn)_(?P<percentile>\d+)"
        r"(?:_(?P<epsilon>\d+(?:\.\d+)?))?"
        r"(?:_(?P<mech>laplace|gaussian))?"
        r"(?:_(?P<label_noise>True|False))?"
        r"_wise\.sav",
        name,
    )
    if dpquery:
        meta = dpquery.groupdict()
        return {
            "privacy_mode": "dpquery",
            "dataset": meta["dataset"],
            "kind": meta["kind"],
            "guiding_bb": meta["bb"],
            "percentile": int(meta["percentile"]),
            "query_epsilon": float(meta["epsilon"]) if meta["epsilon"] is not None else None,
            "query_noise_mech": meta["mech"],
            "noise_on_labeling": meta["label_noise"],
            "model_path": str(model_path),
            "model_name": name,
            "model_case_id": name.replace(".sav", ""),
        }

    dp = re.fullmatch(
        r"rf_dp_(?P<epsilon>\d+(?:\.\d+)?)_(?P<mech>laplace|gaussian)_(?P<dataset>.+)_(?P<kind>entropy|margin|kappa|logit)_(?P<bb>rf|nn)_(?P<percentile>\d+)\.sav",
        name,
    )
    if dp:
        meta = dp.groupdict()
        return {
            "privacy_mode": "dp",
            "dataset": meta["dataset"],
            "kind": meta["kind"],
            "guiding_bb": meta["bb"],
            "percentile": int(meta["percentile"]),
            "query_epsilon": float(meta["epsilon"]),
            "query_noise_mech": meta["mech"],
            "noise_on_labeling": None,
            "model_path": str(model_path),
            "model_name": name,
            "model_case_id": name.replace(".sav", ""),
        }

    return None


def discover_models() -> list[dict]:
    found = []
    for model_path in sorted(DIR_MODELS.glob("*/*.sav")):
        meta = parse_model_filename(model_path)
        if meta is None:
            continue
        if meta["dataset"] not in DATASETS:
            continue
        if meta["kind"] != "unknown" and meta["kind"] != KIND:
            continue
        if meta["guiding_bb"] != "unknown" and meta["guiding_bb"] != GUIDING_BB:
            continue
        if meta["percentile"] is not None and meta["percentile"] != PERCENTILE:
            continue
        if MODEL_SCOPE == "standard" and meta["privacy_mode"] != "standard":
            continue
        if MODEL_SCOPE == "dpquery" and meta["privacy_mode"] != "dpquery":
            continue
        if MODEL_SCOPE == "dp" and meta["privacy_mode"] != "dp":
            continue
        found.append(meta)
    return found


def load_model(model_path: str):
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def get_node_path(tree, x_row: np.ndarray) -> tuple[list[RuleCondition], int]:
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold

    node_id = 0
    conditions: list[RuleCondition] = []

    while children_left[node_id] != children_right[node_id]:
        feature_idx = int(features[node_id])
        threshold = float(thresholds[node_id])

        if x_row[feature_idx] <= threshold:
            conditions.append(RuleCondition(feature_idx, "<=", threshold))
            node_id = int(children_left[node_id])
        else:
            conditions.append(RuleCondition(feature_idx, ">", threshold))
            node_id = int(children_right[node_id])

    return conditions, node_id


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def render_rule(conditions: Iterable[RuleCondition], feature_names: list[str]) -> str:
    parts = []
    for condition in conditions:
        feat = feature_names[condition.feature_idx]
        parts.append(f"{feat} {condition.operator} {condition.threshold:.6f}")
    return " AND ".join(parts) if parts else "[root]"


def extract_supporting_rules(model, x_row: np.ndarray, row_id: int) -> tuple[int, list[RuleRecord]]:
    rf_pred = int(model.predict(x_row.reshape(1, -1))[0])
    rules: list[RuleRecord] = []

    for tree_idx, estimator in enumerate(model.estimators_):
        tree_pred = int(estimator.predict(x_row.reshape(1, -1))[0])
        if tree_pred != rf_pred:
            continue

        tree = estimator.tree_
        conditions, leaf_idx = get_node_path(tree, x_row)

        root_count = float(tree.n_node_samples[0]) if tree.n_node_samples[0] > 0 else 1.0
        leaf_count = float(tree.n_node_samples[leaf_idx])
        leaf_support_train = leaf_count / root_count

        leaf_values = tree.value[leaf_idx][0]
        leaf_total = float(np.sum(leaf_values))
        leaf_purity = float(np.max(leaf_values) / leaf_total) if leaf_total > 0 else 0.0

        features_used = [condition.feature_idx for condition in conditions]
        rule_length = len(conditions)
        score = float(leaf_purity * leaf_support_train / (1.0 + rule_length))

        rules.append(
            RuleRecord(
                row_id=row_id,
                rf_pred=rf_pred,
                tree_idx=tree_idx,
                leaf_idx=int(leaf_idx),
                rule_length=rule_length,
                unique_features=len(set(features_used)),
                leaf_support_train=leaf_support_train,
                leaf_purity=leaf_purity,
                score=score,
                feature_set=set(features_used),
                conditions=conditions,
            )
        )

    return rf_pred, rules


def select_top_k_diverse(rules: list[RuleRecord], k: int) -> list[dict]:
    ranked = sorted(rules, key=lambda rule: (rule.score, rule.leaf_purity, rule.leaf_support_train), reverse=True)

    selected = []
    skipped = []

    for rule in ranked:
        max_jaccard = max(
            (jaccard_similarity(rule.feature_set, prev["rule"].feature_set) for prev in selected),
            default=0.0,
        )
        if len(selected) < k and max_jaccard <= MAX_FEATURE_JACCARD:
            selected.append({
                "rule": rule,
                "selection_reason": "diverse_pass",
                "max_jaccard_to_selected": max_jaccard,
            })
        else:
            skipped.append({
                "rule": rule,
                "selection_reason": "skipped_diversity",
                "max_jaccard_to_selected": max_jaccard,
            })

    if FALLBACK_FILL and len(selected) < k:
        for item in skipped:
            if len(selected) >= k:
                break
            item = dict(item)
            item["selection_reason"] = "fallback_fill"
            selected.append(item)

    selected = selected[:k]
    for rank, item in enumerate(selected, 1):
        item["selected_rank"] = rank
    return selected


def rule_to_row(meta: dict,
                rule: RuleRecord,
                feature_names: list[str],
                n_trees: int) -> dict:
    return {
        "dataset": meta["dataset"],
        "kind": meta["kind"],
        "guiding_bb": meta["guiding_bb"],
        "percentile": meta["percentile"],
        "privacy_mode": meta["privacy_mode"],
        "query_epsilon": meta["query_epsilon"],
        "query_noise_mech": meta["query_noise_mech"],
        "noise_on_labeling": meta["noise_on_labeling"],
        "model_name": meta["model_name"],
        "model_path": meta["model_path"],
        "model_case_id": meta["model_case_id"],
        "row_id": rule.row_id,
        "rf_pred": rule.rf_pred,
        "n_trees": n_trees,
        "tree_idx": rule.tree_idx,
        "leaf_idx": rule.leaf_idx,
        "rule_length": rule.rule_length,
        "unique_features": rule.unique_features,
        "leaf_support_train": rule.leaf_support_train,
        "leaf_purity": rule.leaf_purity,
        "score": rule.score,
        "feature_indices": json.dumps(sorted(rule.feature_set)),
        "conditions_json": json.dumps(
            [
                {
                    "feature_idx": cond.feature_idx,
                    "operator": cond.operator,
                    "threshold": cond.threshold,
                    "feature_name": feature_names[cond.feature_idx],
                }
                for cond in rule.conditions
            ]
        ),
        "rendered_rule": render_rule(rule.conditions, feature_names) if SAVE_RENDERED_RULES else "",
    }


def summarize_selection(meta: dict,
                        row_id: int,
                        rf_pred: int,
                        n_trees: int,
                        all_rules: list[RuleRecord],
                        selected_items: list[dict],
                        k: int) -> dict:
    all_scores = [rule.score for rule in all_rules]
    selected_rules = [item["rule"] for item in selected_items]
    selected_scores = [rule.score for rule in selected_rules]

    all_total_literals = int(sum(rule.rule_length for rule in all_rules))
    selected_total_literals = int(sum(rule.rule_length for rule in selected_rules))

    all_features = set().union(*(rule.feature_set for rule in all_rules)) if all_rules else set()
    selected_features = set().union(*(rule.feature_set for rule in selected_rules)) if selected_rules else set()

    return {
        "dataset": meta["dataset"],
        "kind": meta["kind"],
        "guiding_bb": meta["guiding_bb"],
        "percentile": meta["percentile"],
        "privacy_mode": meta["privacy_mode"],
        "query_epsilon": meta["query_epsilon"],
        "query_noise_mech": meta["query_noise_mech"],
        "noise_on_labeling": meta["noise_on_labeling"],
        "model_name": meta["model_name"],
        "model_path": meta["model_path"],
        "model_case_id": meta["model_case_id"],
        "row_id": row_id,
        "rf_pred": rf_pred,
        "n_trees": n_trees,
        "target_k": k,
        "n_supporting_rules_full": len(all_rules),
        "n_supporting_rules_selected": len(selected_rules),
        "agreement_ratio_full": float(len(all_rules) / n_trees) if n_trees > 0 else 0.0,
        "agreement_ratio_selected": float(len(selected_rules) / n_trees) if n_trees > 0 else 0.0,
        "total_literals_full": all_total_literals,
        "total_literals_selected": selected_total_literals,
        "literal_reduction_ratio": float(1.0 - (selected_total_literals / all_total_literals)) if all_total_literals > 0 else 0.0,
        "unique_features_full": len(all_features),
        "unique_features_selected": len(selected_features),
        "score_sum_full": float(np.sum(all_scores)) if all_scores else 0.0,
        "score_sum_selected": float(np.sum(selected_scores)) if selected_scores else 0.0,
        "score_retention_ratio": float(np.sum(selected_scores) / np.sum(all_scores)) if np.sum(all_scores) > 0 else 0.0,
        "mean_score_full": float(np.mean(all_scores)) if all_scores else 0.0,
        "mean_score_selected": float(np.mean(selected_scores)) if selected_scores else 0.0,
    }


def evaluate_model(meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_ref, feature_names = load_reference_data(meta["dataset"], split=EVAL_SPLIT)
    model = load_model(meta["model_path"])
    n_trees = len(model.estimators_)

    all_rule_rows = []
    selected_rule_rows = []
    instance_summary_rows = []

    for row_id, x_row in enumerate(x_ref):
        rf_pred, rules = extract_supporting_rules(model, x_row, row_id=row_id)

        for rule in rules:
            all_rule_rows.append(rule_to_row(meta, rule, feature_names, n_trees))

        for k in K_VALUES:
            selected_items = select_top_k_diverse(rules, k)

            for item in selected_items:
                rule = item["rule"]
                row = rule_to_row(meta, rule, feature_names, n_trees)
                row.update({
                    "target_k": k,
                    "selected_rank": item["selected_rank"],
                    "selection_reason": item["selection_reason"],
                    "max_jaccard_to_selected": item["max_jaccard_to_selected"],
                })
                selected_rule_rows.append(row)

            instance_summary_rows.append(
                summarize_selection(
                    meta=meta,
                    row_id=row_id,
                    rf_pred=rf_pred,
                    n_trees=n_trees,
                    all_rules=rules,
                    selected_items=selected_items,
                    k=k,
                )
            )

    return (
        pd.DataFrame(all_rule_rows),
        pd.DataFrame(selected_rule_rows),
        pd.DataFrame(instance_summary_rows),
    )


def aggregate_instance_summary(instance_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "kind", "guiding_bb", "percentile",
        "privacy_mode", "query_epsilon", "query_noise_mech", "noise_on_labeling",
        "model_name", "model_path", "target_k",
    ]

    rows = []
    for keys, chunk in instance_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update({
            "n_instances": int(len(chunk)),
            "avg_supporting_rules_full": float(chunk["n_supporting_rules_full"].mean()),
            "avg_supporting_rules_selected": float(chunk["n_supporting_rules_selected"].mean()),
            "avg_total_literals_full": float(chunk["total_literals_full"].mean()),
            "avg_total_literals_selected": float(chunk["total_literals_selected"].mean()),
            "avg_literal_reduction_ratio": float(chunk["literal_reduction_ratio"].mean()),
            "avg_unique_features_full": float(chunk["unique_features_full"].mean()),
            "avg_unique_features_selected": float(chunk["unique_features_selected"].mean()),
            "avg_score_retention_ratio": float(chunk["score_retention_ratio"].mean()),
            "avg_mean_score_full": float(chunk["mean_score_full"].mean()),
            "avg_mean_score_selected": float(chunk["mean_score_selected"].mean()),
            "avg_agreement_ratio_full": float(chunk["agreement_ratio_full"].mean()),
            "avg_agreement_ratio_selected": float(chunk["agreement_ratio_selected"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ensure_dir(DIR_REPORTS)

    models = discover_models()
    if not models:
        raise RuntimeError("No compatible model found for the current configuration.")

    metadata_path = DIR_REPORTS / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "kind": KIND,
                "guiding_bb": GUIDING_BB,
                "percentile": PERCENTILE,
                "model_scope": MODEL_SCOPE,
                "eval_split": EVAL_SPLIT,
                "max_eval_rows": MAX_EVAL_ROWS,
                "k_values": K_VALUES,
                "max_feature_jaccard": MAX_FEATURE_JACCARD,
                "fallback_fill": FALLBACK_FILL,
                "n_models_found": len(models),
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_rules_frames = []
    selected_rules_frames = []
    instance_summary_frames = []

    for meta in models:
        print(f"\n=== Processing {meta['model_name']} ===")
        try:
            all_df, selected_df, instance_df = evaluate_model(meta)
            all_rules_frames.append(all_df)
            selected_rules_frames.append(selected_df)
            instance_summary_frames.append(instance_df)

            dataset_dir = DIR_REPORTS / meta["dataset"] / meta["model_case_id"]
            ensure_dir(dataset_dir)

            model_metadata_path = dataset_dir / "model_metadata.json"
            model_metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            all_df.to_csv(dataset_dir / "all_supporting_rules.csv", index=False)
            selected_df.to_csv(dataset_dir / "selected_supporting_rules.csv", index=False)
            instance_df.to_csv(dataset_dir / "instance_selection_summary.csv", index=False)

            aggregate_df_model = aggregate_instance_summary(instance_df) if not instance_df.empty else pd.DataFrame()
            aggregate_df_model.to_csv(dataset_dir / "aggregate_selection_summary.csv", index=False)

            print(
                f"[ok] {meta['model_name']}: "
                f"all_rules={len(all_df)}, selected_rules={len(selected_df)}, "
                f"instance_rows={len(instance_df)}"
            )
        except Exception as exc:
            print(f"[err] {meta['model_name']}: {type(exc).__name__}: {exc}")

    if not all_rules_frames:
        raise RuntimeError("No result produced.")

    all_rules_df = pd.concat(all_rules_frames, ignore_index=True)
    selected_rules_df = pd.concat(selected_rules_frames, ignore_index=True) if selected_rules_frames else pd.DataFrame()
    instance_summary_df = pd.concat(instance_summary_frames, ignore_index=True) if instance_summary_frames else pd.DataFrame()
    aggregate_df = aggregate_instance_summary(instance_summary_df) if not instance_summary_df.empty else pd.DataFrame()

    all_rules_path = DIR_REPORTS / "all_supporting_rules.csv"
    selected_rules_path = DIR_REPORTS / "selected_supporting_rules.csv"
    instance_summary_path = DIR_REPORTS / "instance_selection_summary.csv"
    aggregate_path = DIR_REPORTS / "aggregate_selection_summary.csv"

    all_rules_df.to_csv(all_rules_path, index=False)
    selected_rules_df.to_csv(selected_rules_path, index=False)
    instance_summary_df.to_csv(instance_summary_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print("\n[done] Saved:")
    print(f"  - {all_rules_path}")
    print(f"  - {selected_rules_path}")
    print(f"  - {instance_summary_path}")
    print(f"  - {aggregate_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()
