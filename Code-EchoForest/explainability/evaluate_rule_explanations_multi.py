#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate local rule-based explanations extracted from PREMS Random Forest surrogates.

"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

DIR_ORIGINAL = DATA_ORIGINAL_DIR
DIR_MODELS = MODEL_SYNTHETIC_WISE_DIR
DIR_REPORTS = REPORTS_EVAL_DIR / "rule-explanations-logit-nn-25"

RANDOM_STATE = 42
EVAL_SPLIT = "test"
MAX_EVAL_ROWS = 5000
N_EXPLANATION_EXAMPLES = 10


@dataclass
class RuleCondition:
    feature_idx: int
    operator: str
    threshold: float


@dataclass
class RuleSummary:
    tree_idx: int
    leaf_idx: int
    rule_length: int
    unique_features: int
    coverage_ref: float
    leaf_support_train: float
    leaf_purity: float
    conditions: list[RuleCondition]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model(dataset: str):
    model_path = DIR_MODELS / dataset / f"rf_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}_wise.sav"
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, model_path


def load_reference_data(dataset: str, split: str = EVAL_SPLIT) -> tuple[np.ndarray, list[str], pd.DataFrame]:
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
        ref_df = train_df.reset_index(drop=True)
    elif split == "all":
        x_ref = np.vstack([x_train, x_test])
        ref_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    else:
        x_ref = x_test
        ref_df = test_df.reset_index(drop=True)

    if len(x_ref) > MAX_EVAL_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        chosen = np.sort(rng.choice(len(x_ref), size=MAX_EVAL_ROWS, replace=False))
        x_ref = x_ref[chosen]
        ref_df = ref_df.iloc[chosen].reset_index(drop=True)

    return x_ref.astype(np.float32), feature_names, ref_df


def get_node_path(tree, x_row: np.ndarray) -> tuple[list[int], list[RuleCondition], int]:
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold

    node_id = 0
    node_path = [0]
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

        node_path.append(node_id)

    return node_path, conditions, node_id


def apply_conditions(x_ref: np.ndarray, conditions: Iterable[RuleCondition]) -> np.ndarray:
    mask = np.ones(len(x_ref), dtype=bool)
    for condition in conditions:
        if condition.operator == "<=":
            mask &= x_ref[:, condition.feature_idx] <= condition.threshold
        else:
            mask &= x_ref[:, condition.feature_idx] > condition.threshold
    return mask


def extract_supporting_rules(model, x_row: np.ndarray, x_ref: np.ndarray) -> tuple[int, list[RuleSummary]]:
    rf_pred = int(model.predict(x_row.reshape(1, -1))[0])
    rules: list[RuleSummary] = []

    for tree_idx, estimator in enumerate(model.estimators_):
        tree_pred = int(estimator.predict(x_row.reshape(1, -1))[0])
        if tree_pred != rf_pred:
            continue

        tree = estimator.tree_
        _, conditions, leaf_idx = get_node_path(tree, x_row)

        coverage_mask = apply_conditions(x_ref, conditions)
        coverage_ref = float(np.mean(coverage_mask)) if len(coverage_mask) else 0.0

        root_count = float(tree.n_node_samples[0]) if tree.n_node_samples[0] > 0 else 1.0
        leaf_count = float(tree.n_node_samples[leaf_idx])
        leaf_support_train = leaf_count / root_count

        leaf_values = tree.value[leaf_idx][0]
        leaf_total = float(np.sum(leaf_values))
        leaf_purity = float(np.max(leaf_values) / leaf_total) if leaf_total > 0 else 0.0

        features_used = [condition.feature_idx for condition in conditions]

        rules.append(
            RuleSummary(
                tree_idx=tree_idx,
                leaf_idx=int(leaf_idx),
                rule_length=len(conditions),
                unique_features=len(set(features_used)),
                coverage_ref=coverage_ref,
                leaf_support_train=leaf_support_train,
                leaf_purity=leaf_purity,
                conditions=conditions,
            )
        )

    return rf_pred, rules


def format_rule(rule: RuleSummary, feature_names: list[str]) -> str:
    if not rule.conditions:
        return "[root]"

    parts = []
    for condition in rule.conditions:
        feature_name = feature_names[condition.feature_idx]
        parts.append(f"{feature_name} {condition.operator} {condition.threshold:.4f}")
    return " AND ".join(parts)


def summarize_instance(model,
                       x_row: np.ndarray,
                       x_ref: np.ndarray,
                       feature_names: list[str],
                       row_id: int) -> tuple[dict, list[str]]:
    rf_pred, rules = extract_supporting_rules(model, x_row, x_ref)
    n_trees = len(model.estimators_)
    n_supporting = len(rules)

    if n_supporting == 0:
        return {
            "row_id": row_id,
            "rf_pred": rf_pred,
            "n_trees": n_trees,
            "n_supporting_rules": 0,
            "agreement_ratio": 0.0,
            "total_literals": 0,
            "mean_rule_length": 0.0,
            "median_rule_length": 0.0,
            "max_rule_length": 0,
            "unique_features_total": 0,
            "feature_reuse_ratio": 0.0,
            "mean_rule_coverage": 0.0,
            "min_rule_coverage": 0.0,
            "max_rule_coverage": 0.0,
            "mean_leaf_support_train": 0.0,
            "mean_leaf_purity": 0.0,
        }, []

    rule_lengths = np.array([rule.rule_length for rule in rules], dtype=float)
    rule_coverages = np.array([rule.coverage_ref for rule in rules], dtype=float)
    leaf_supports = np.array([rule.leaf_support_train for rule in rules], dtype=float)
    leaf_purities = np.array([rule.leaf_purity for rule in rules], dtype=float)

    all_features = [condition.feature_idx for rule in rules for condition in rule.conditions]
    unique_features_total = len(set(all_features))
    total_literals = int(np.sum(rule_lengths))

    feature_reuse_ratio = float(total_literals / unique_features_total) if unique_features_total > 0 else 0.0

    row = {
        "row_id": row_id,
        "rf_pred": rf_pred,
        "n_trees": n_trees,
        "n_supporting_rules": n_supporting,
        "agreement_ratio": float(n_supporting / n_trees),
        "total_literals": total_literals,
        "mean_rule_length": float(np.mean(rule_lengths)),
        "median_rule_length": float(np.median(rule_lengths)),
        "max_rule_length": int(np.max(rule_lengths)),
        "unique_features_total": unique_features_total,
        "feature_reuse_ratio": feature_reuse_ratio,
        "mean_rule_coverage": float(np.mean(rule_coverages)),
        "min_rule_coverage": float(np.min(rule_coverages)),
        "max_rule_coverage": float(np.max(rule_coverages)),
        "mean_leaf_support_train": float(np.mean(leaf_supports)),
        "mean_leaf_purity": float(np.mean(leaf_purities)),
    }

    rendered_rules = [format_rule(rule, feature_names) for rule in rules]
    return row, rendered_rules


def aggregate_instance_metrics(df: pd.DataFrame) -> dict:
    out = {
        "n_instances": int(len(df)),
        "avg_supporting_rules": float(df["n_supporting_rules"].mean()),
        "median_supporting_rules": float(df["n_supporting_rules"].median()),
        "p90_supporting_rules": float(df["n_supporting_rules"].quantile(0.90)),
        "avg_total_literals": float(df["total_literals"].mean()),
        "median_total_literals": float(df["total_literals"].median()),
        "p90_total_literals": float(df["total_literals"].quantile(0.90)),
        "avg_rule_length": float(df["mean_rule_length"].mean()),
        "median_rule_length": float(df["mean_rule_length"].median()),
        "avg_unique_features_total": float(df["unique_features_total"].mean()),
        "avg_feature_reuse_ratio": float(df["feature_reuse_ratio"].mean()),
        "avg_agreement_ratio": float(df["agreement_ratio"].mean()),
        "avg_rule_coverage": float(df["mean_rule_coverage"].mean()),
        "median_rule_coverage": float(df["mean_rule_coverage"].median()),
        "avg_leaf_support_train": float(df["mean_leaf_support_train"].mean()),
        "avg_leaf_purity": float(df["mean_leaf_purity"].mean()),
    }
    return out


def save_example_explanations(dataset: str,
                              ref_df: pd.DataFrame,
                              instance_rows: pd.DataFrame,
                              explanations: dict[int, list[str]],
                              out_path: Path) -> None:
    lines = []
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Examples shown: {min(N_EXPLANATION_EXAMPLES, len(instance_rows))}")
    lines.append("")

    for _, row in instance_rows.head(N_EXPLANATION_EXAMPLES).iterrows():
        row_id = int(row["row_id"])
        lines.append(f"[row_id={row_id}] pred={row['rf_pred']} supporting_rules={row['n_supporting_rules']}")
        lines.append(
            "metrics: "
            f"agreement={row['agreement_ratio']:.4f}, "
            f"total_literals={int(row['total_literals'])}, "
            f"mean_rule_length={row['mean_rule_length']:.2f}, "
            f"unique_features={int(row['unique_features_total'])}"
        )
        if row_id < len(ref_df):
            lines.append("original_features_sample:")
            preview = ref_df.iloc[row_id].to_dict()
            for key, value in list(preview.items())[:12]:
                lines.append(f"  - {key}: {value}")
        lines.append("rules:")
        for idx, rendered in enumerate(explanations.get(row_id, [])[:15], 1):
            lines.append(f"  {idx}. {rendered}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_dataset(dataset: str) -> tuple[dict, pd.DataFrame]:
    model, model_path = load_model(dataset)
    x_ref, feature_names, ref_df = load_reference_data(dataset, split=EVAL_SPLIT)

    instance_rows = []
    explanations: dict[int, list[str]] = {}

    for row_id, x_row in enumerate(x_ref):
        instance_row, rendered_rules = summarize_instance(
            model=model,
            x_row=x_row,
            x_ref=x_ref,
            feature_names=feature_names,
            row_id=row_id,
        )
        instance_rows.append(instance_row)
        explanations[row_id] = rendered_rules

    instance_df = pd.DataFrame(instance_rows)
    aggregate = aggregate_instance_metrics(instance_df)
    aggregate.update({
        "dataset": dataset,
        "model_path": str(model_path),
        "split": EVAL_SPLIT,
        "eval_rows": int(len(instance_df)),
        "n_trees": int(len(model.estimators_)),
    })

    out_dataset_dir = DIR_REPORTS / dataset
    ensure_dir(out_dataset_dir)

    instance_path = out_dataset_dir / f"rule_metrics_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}.csv"
    instance_df.to_csv(instance_path, index=False)

    examples_path = out_dataset_dir / f"rule_examples_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}.txt"
    save_example_explanations(dataset, ref_df, instance_df, explanations, examples_path)

    aggregate["instance_metrics_path"] = str(instance_path)
    aggregate["example_rules_path"] = str(examples_path)

    return aggregate, instance_df


def main():
    ensure_dir(DIR_REPORTS)

    aggregate_rows = []
    for dataset in DATASETS:
        print(f"\n=== Evaluating rule explanations for {dataset} ===")
        try:
            aggregate_row, _ = evaluate_dataset(dataset)
            aggregate_rows.append(aggregate_row)
            print(
                f"[ok] {dataset}: avg_supporting_rules={aggregate_row['avg_supporting_rules']:.2f}, "
                f"avg_total_literals={aggregate_row['avg_total_literals']:.2f}, "
                f"avg_rule_length={aggregate_row['avg_rule_length']:.2f}"
            )
        except Exception as exc:
            print(f"[err] {dataset}: {type(exc).__name__}: {exc}")
            aggregate_rows.append({
                "dataset": dataset,
                "error": f"{type(exc).__name__}: {exc}",
            })

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = DIR_REPORTS / f"rule_metrics_summary_{KIND}_{GUIDING_BB}_{PERCENTILE}.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"\n[done] Summary salvata in: {aggregate_path}")


if __name__ == "__main__":
    main()
