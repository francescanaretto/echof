#!/usr/bin/env python3
"""
Analyze PREMS explanations on the exact instances already explained by LIME/SHAP.

"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from shared.project_paths import MODEL_SYNTHETIC_WISE_DIR, REPORTS_EVAL_DIR, PROJECT_ROOT

from explainability.select_supporting_rules import (
    load_model,
    extract_supporting_rules,
    select_top_k_diverse,
    rule_to_row,
)


ROOT = PROJECT_ROOT
DIR_POSTHOC = REPORTS_EVAL_DIR / "posthoc-explainers"
DIR_OUT = DIR_POSTHOC
DIR_MODELS = MODEL_SYNTHETIC_WISE_DIR

KIND = "logit"
GUIDING_BB = "nn"
PERCENTILE = 25
TARGET_K = 8


def model_path_for_dataset(dataset: str) -> Path:
    return DIR_MODELS / dataset / f"rf_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}_wise.sav"


def load_full_test_reference_data(dataset: str) -> tuple[np.ndarray, list[str]]:
    """
    Load the full standardized test split without any subsampling.

    """
    base = ROOT / "Data-original" / dataset
    train_df = pd.read_csv(base / f"train_set_{dataset}.csv")
    test_df = pd.read_csv(base / f"test_set_{dataset}.csv")

    train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.startswith("Unnamed")]

    feature_names = train_df.columns.tolist()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df.values.astype(np.float32))
    x_test = scaler.transform(test_df.values.astype(np.float32))

    return x_test.astype(np.float32), feature_names


def load_posthoc_instance_ids() -> dict[str, list[int]]:
    path = DIR_POSTHOC / "lime_shap_explanations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing post-hoc explanations file: {path}")

    df = pd.read_csv(path)
    needed = {"dataset", "instance_id"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Expected columns {needed} in {path}")

    grouped = {}
    for dataset, chunk in df.groupby("dataset"):
        ids = sorted(int(v) for v in chunk["instance_id"].dropna().unique().tolist())
        grouped[dataset] = ids
    return grouped


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize_feature_overlap(selected_rules: list, top_features: set[int]) -> tuple[int, float]:
    selected_features = set().union(*(rule.feature_set for rule in selected_rules)) if selected_rules else set()
    if not top_features and not selected_features:
        return 0, 1.0
    union = top_features | selected_features
    if not union:
        return 0, 0.0
    inter = top_features & selected_features
    return len(inter), float(len(inter) / len(union))


def build_meta(dataset: str, model_path: Path) -> dict:
    return {
        "dataset": dataset,
        "kind": KIND,
        "guiding_bb": GUIDING_BB,
        "percentile": PERCENTILE,
        "privacy_mode": "standard",
        "query_epsilon": None,
        "query_noise_mech": None,
        "noise_on_labeling": None,
        "model_name": model_path.name,
        "model_path": str(model_path),
        "model_case_id": model_path.stem,
    }


def main() -> None:
    instance_ids_by_dataset = load_posthoc_instance_ids()
    all_rows = []
    selected_rule_rows = []

    for dataset in sorted(instance_ids_by_dataset):
        ids = instance_ids_by_dataset[dataset]
        model_path = model_path_for_dataset(dataset)
        if not model_path.exists():
            print(f"[skip] model not found for {dataset}: {model_path}")
            continue

        x_ref, feature_names = load_full_test_reference_data(dataset)
        model = load_model(str(model_path))
        n_trees = len(model.estimators_)
        meta = build_meta(dataset, model_path)

        valid_ids = [idx for idx in ids if 0 <= idx < len(x_ref)]
        if len(valid_ids) != len(ids):
            print(f"[warn] dataset={dataset}: kept {len(valid_ids)}/{len(ids)} valid test indices")

        print(f"[info] dataset={dataset}: {len(valid_ids)} instances")

        for row_id in valid_ids:
            x_row = x_ref[row_id]

            t0 = time.perf_counter()
            rf_pred, all_rules = extract_supporting_rules(model, x_row, row_id=row_id)
            extraction_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            selected_items = select_top_k_diverse(all_rules, TARGET_K)
            selection_time = time.perf_counter() - t1

            selected_rules = [item["rule"] for item in selected_items]

            all_rule_lengths = [rule.rule_length for rule in all_rules]
            selected_rule_lengths = [rule.rule_length for rule in selected_rules]
            all_supports = [rule.leaf_support_train for rule in all_rules]
            selected_supports = [rule.leaf_support_train for rule in selected_rules]
            all_purities = [rule.leaf_purity for rule in all_rules]
            selected_purities = [rule.leaf_purity for rule in selected_rules]
            all_scores = [rule.score for rule in all_rules]
            selected_scores = [rule.score for rule in selected_rules]

            all_features = set().union(*(rule.feature_set for rule in all_rules)) if all_rules else set()
            selected_features = set().union(*(rule.feature_set for rule in selected_rules)) if selected_rules else set()

            all_rows.append(
                {
                    "dataset": dataset,
                    "instance_id": int(row_id),
                    "rf_pred": int(rf_pred),
                    "n_trees": int(n_trees),
                    "target_k": TARGET_K,
                    "n_rules_all": int(len(all_rules)),
                    "n_rules_selected": int(len(selected_rules)),
                    "rules_reduction_pct": float(100.0 * (1.0 - (len(selected_rules) / len(all_rules)))) if all_rules else 0.0,
                    "literals_all": int(sum(all_rule_lengths)),
                    "literals_selected": int(sum(selected_rule_lengths)),
                    "literals_reduction_pct": float(
                        100.0 * (1.0 - (sum(selected_rule_lengths) / sum(all_rule_lengths)))
                    ) if sum(all_rule_lengths) > 0 else 0.0,
                    "mean_rule_length_all": safe_mean(all_rule_lengths),
                    "mean_rule_length_selected": safe_mean(selected_rule_lengths),
                    "unique_features_all": int(len(all_features)),
                    "unique_features_selected": int(len(selected_features)),
                    "mean_leaf_support_all": safe_mean(all_supports),
                    "mean_leaf_support_selected": safe_mean(selected_supports),
                    "mean_leaf_purity_all": safe_mean(all_purities),
                    "mean_leaf_purity_selected": safe_mean(selected_purities),
                    "mean_score_all": safe_mean(all_scores),
                    "mean_score_selected": safe_mean(selected_scores),
                    "score_sum_all": float(np.sum(all_scores)) if all_scores else 0.0,
                    "score_sum_selected": float(np.sum(selected_scores)) if selected_scores else 0.0,
                    "score_retention_ratio": float(np.sum(selected_scores) / np.sum(all_scores)) if np.sum(all_scores) > 0 else 0.0,
                    "agreement_ratio_all": float(len(all_rules) / n_trees) if n_trees > 0 else 0.0,
                    "agreement_ratio_selected": float(len(selected_rules) / n_trees) if n_trees > 0 else 0.0,
                    "time_extract_seconds": float(extraction_time),
                    "time_select_seconds": float(selection_time),
                    "time_total_seconds": float(extraction_time + selection_time),
                }
            )

            for item in selected_items:
                row = rule_to_row(meta, item["rule"], feature_names, n_trees)
                row.update(
                    {
                        "target_k": TARGET_K,
                        "selected_rank": item["selected_rank"],
                        "selection_reason": item["selection_reason"],
                        "max_jaccard_to_selected": item["max_jaccard_to_selected"],
                    }
                )
                selected_rule_rows.append(row)

    if not all_rows:
        raise RuntimeError("No PREMS rows computed from post-hoc instances.")

    per_instance_df = pd.DataFrame(all_rows)
    selected_rules_df = pd.DataFrame(selected_rule_rows)

    summary_by_dataset = (
        per_instance_df.groupby("dataset", as_index=False)
        .agg(
            n_instances=("instance_id", "nunique"),
            n_rules_all_mean=("n_rules_all", "mean"),
            n_rules_all_median=("n_rules_all", "median"),
            n_rules_selected_mean=("n_rules_selected", "mean"),
            rules_reduction_pct_mean=("rules_reduction_pct", "mean"),
            literals_all_mean=("literals_all", "mean"),
            literals_selected_mean=("literals_selected", "mean"),
            literals_reduction_pct_mean=("literals_reduction_pct", "mean"),
            mean_rule_length_all=("mean_rule_length_all", "mean"),
            mean_rule_length_selected=("mean_rule_length_selected", "mean"),
            unique_features_all_mean=("unique_features_all", "mean"),
            unique_features_selected_mean=("unique_features_selected", "mean"),
            mean_leaf_support_all=("mean_leaf_support_all", "mean"),
            mean_leaf_support_selected=("mean_leaf_support_selected", "mean"),
            mean_leaf_purity_all=("mean_leaf_purity_all", "mean"),
            mean_leaf_purity_selected=("mean_leaf_purity_selected", "mean"),
            mean_score_all=("mean_score_all", "mean"),
            mean_score_selected=("mean_score_selected", "mean"),
            score_retention_ratio_mean=("score_retention_ratio", "mean"),
            time_extract_seconds_mean=("time_extract_seconds", "mean"),
            time_extract_seconds_std=("time_extract_seconds", "std"),
            time_select_seconds_mean=("time_select_seconds", "mean"),
            time_select_seconds_std=("time_select_seconds", "std"),
            time_total_seconds_mean=("time_total_seconds", "mean"),
            time_total_seconds_std=("time_total_seconds", "std"),
        )
        .sort_values("dataset")
    )

    global_row = {
        "n_datasets": int(summary_by_dataset["dataset"].nunique()),
        "n_instances_total": int(per_instance_df["instance_id"].count()),
        "rules_reduction_pct_mean": float(per_instance_df["rules_reduction_pct"].mean()),
        "literals_reduction_pct_mean": float(per_instance_df["literals_reduction_pct"].mean()),
        "mean_leaf_purity_all": float(per_instance_df["mean_leaf_purity_all"].mean()),
        "mean_leaf_purity_selected": float(per_instance_df["mean_leaf_purity_selected"].mean()),
        "mean_leaf_support_all": float(per_instance_df["mean_leaf_support_all"].mean()),
        "mean_leaf_support_selected": float(per_instance_df["mean_leaf_support_selected"].mean()),
        "time_extract_seconds_mean": float(per_instance_df["time_extract_seconds"].mean()),
        "time_select_seconds_mean": float(per_instance_df["time_select_seconds"].mean()),
        "time_total_seconds_mean": float(per_instance_df["time_total_seconds"].mean()),
    }
    global_df = pd.DataFrame([global_row])

    per_instance_path = DIR_OUT / "prems_on_posthoc_instances_per_instance.csv"
    summary_path = DIR_OUT / "prems_on_posthoc_instances_summary_by_dataset.csv"
    global_path = DIR_OUT / "prems_on_posthoc_instances_summary_global.csv"
    selected_rules_path = DIR_OUT / "prems_on_posthoc_instances_selected_rules.csv"
    meta_path = DIR_OUT / "prems_on_posthoc_instances_metadata.json"

    per_instance_df.to_csv(per_instance_path, index=False)
    summary_by_dataset.to_csv(summary_path, index=False)
    global_df.to_csv(global_path, index=False)
    selected_rules_df.to_csv(selected_rules_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "kind": KIND,
                "guiding_bb": GUIDING_BB,
                "percentile": PERCENTILE,
                "target_k": TARGET_K,
                "datasets": sorted(summary_by_dataset["dataset"].tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote {per_instance_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {global_path}")
    print(f"[OK] wrote {selected_rules_path}")


if __name__ == "__main__":
    main()
