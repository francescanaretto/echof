#!/usr/bin/env python3
"""
Create compact summaries from Reports-eval/supporting-rule-selection/all_supporting_rules.csv.

"""

from __future__ import annotations

import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "Reports-eval" / "supporting-rule-selection" / "all_supporting_rules.csv"
OUTPUT_DIR = ROOT / "Reports-eval" / "supporting-rule-selection-summary"
TOP_K = 20


def safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def safe_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


@dataclass
class RunningStats:
    count: int = 0
    sum_rule_length: float = 0.0
    sum_unique_features: float = 0.0
    sum_leaf_support: float = 0.0
    sum_leaf_purity: float = 0.0
    sum_score: float = 0.0
    max_score: float = float("-inf")
    max_rule_length: int = 0
    rule_lengths: list[int] = field(default_factory=list)

    def add(self, rule_length: int, unique_features: int, leaf_support: float, leaf_purity: float, score: float) -> None:
        self.count += 1
        self.sum_rule_length += rule_length
        self.sum_unique_features += unique_features
        self.sum_leaf_support += leaf_support
        self.sum_leaf_purity += leaf_purity
        self.sum_score += score
        self.max_score = max(self.max_score, score)
        self.max_rule_length = max(self.max_rule_length, rule_length)
        self.rule_lengths.append(rule_length)

    def mean_rule_length(self) -> float:
        return self.sum_rule_length / self.count if self.count else 0.0

    def mean_unique_features(self) -> float:
        return self.sum_unique_features / self.count if self.count else 0.0

    def mean_leaf_support(self) -> float:
        return self.sum_leaf_support / self.count if self.count else 0.0

    def mean_leaf_purity(self) -> float:
        return self.sum_leaf_purity / self.count if self.count else 0.0

    def mean_score(self) -> float:
        return self.sum_score / self.count if self.count else 0.0

    def median_rule_length(self) -> float:
        return float(median(self.rule_lengths)) if self.rule_lengths else 0.0


@dataclass
class InstanceAccumulator:
    count_rules: int = 0
    total_rule_length: int = 0
    scores: list[float] = field(default_factory=list)

    def add(self, rule_length: int, score: float) -> None:
        self.count_rules += 1
        self.total_rule_length += rule_length
        self.scores.append(score)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_model: dict[tuple[str, str], RunningStats] = {}
    per_dataset: dict[str, RunningStats] = {}
    per_instance: dict[tuple[str, str, str], InstanceAccumulator] = defaultdict(InstanceAccumulator)
    model_meta: dict[tuple[str, str], dict[str, str]] = {}
    top_rules: dict[tuple[str, str], list[tuple[float, int, dict[str, str]]]] = defaultdict(list)

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dataset = row["dataset"]
            model_case_id = row["model_case_id"]
            row_id = row["row_id"]
            rule_length = safe_int(row["rule_length"])
            unique_features = safe_int(row["unique_features"])
            leaf_support = safe_float(row["leaf_support_train"])
            leaf_purity = safe_float(row["leaf_purity"])
            score = safe_float(row["score"])

            model_key = (dataset, model_case_id)
            if model_key not in per_model:
                per_model[model_key] = RunningStats()
                model_meta[model_key] = {
                    "dataset": dataset,
                    "kind": row["kind"],
                    "guiding_bb": row["guiding_bb"],
                    "percentile": row["percentile"],
                    "privacy_mode": row["privacy_mode"],
                    "query_epsilon": row["query_epsilon"],
                    "query_noise_mech": row["query_noise_mech"],
                    "noise_on_labeling": row["noise_on_labeling"],
                    "model_name": row["model_name"],
                    "model_case_id": model_case_id,
                }

            per_model[model_key].add(rule_length, unique_features, leaf_support, leaf_purity, score)
            per_dataset.setdefault(dataset, RunningStats()).add(
                rule_length, unique_features, leaf_support, leaf_purity, score
            )
            per_instance[(dataset, model_case_id, row_id)].add(rule_length, score)

            heap = top_rules[model_key]
            top_entry = {
                "dataset": dataset,
                "kind": row["kind"],
                "guiding_bb": row["guiding_bb"],
                "percentile": row["percentile"],
                "privacy_mode": row["privacy_mode"],
                "query_epsilon": row["query_epsilon"],
                "query_noise_mech": row["query_noise_mech"],
                "noise_on_labeling": row["noise_on_labeling"],
                "model_case_id": model_case_id,
                "row_id": row_id,
                "tree_idx": row["tree_idx"],
                "leaf_idx": row["leaf_idx"],
                "rule_length": row["rule_length"],
                "unique_features": row["unique_features"],
                "leaf_support_train": row["leaf_support_train"],
                "leaf_purity": row["leaf_purity"],
                "score": row["score"],
                "rendered_rule": row["rendered_rule"],
            }
            item = (score, idx, top_entry)
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    per_model_rows: list[dict] = []
    for key, stats in sorted(per_model.items()):
        meta = model_meta[key]
        dataset, model_case_id = key
        instance_keys = [k for k in per_instance if k[0] == dataset and k[1] == model_case_id]
        rules_per_instance = [per_instance[k].count_rules for k in instance_keys]
        literals_per_instance = [per_instance[k].total_rule_length for k in instance_keys]
        mean_rules_per_instance = sum(rules_per_instance) / len(rules_per_instance) if rules_per_instance else 0.0
        mean_literals_per_instance = sum(literals_per_instance) / len(literals_per_instance) if literals_per_instance else 0.0

        per_model_rows.append(
            {
                **meta,
                "n_rules_total": stats.count,
                "n_instances": len(instance_keys),
                "mean_rules_per_instance": f"{mean_rules_per_instance:.4f}",
                "mean_literals_per_instance": f"{mean_literals_per_instance:.4f}",
                "mean_rule_length": f"{stats.mean_rule_length():.4f}",
                "median_rule_length": f"{stats.median_rule_length():.4f}",
                "max_rule_length": stats.max_rule_length,
                "mean_unique_features": f"{stats.mean_unique_features():.4f}",
                "mean_leaf_support_train": f"{stats.mean_leaf_support():.6f}",
                "mean_leaf_purity": f"{stats.mean_leaf_purity():.6f}",
                "mean_score": f"{stats.mean_score():.8f}",
                "max_score": f"{stats.max_score:.8f}",
            }
        )

    per_dataset_rows: list[dict] = []
    for dataset, stats in sorted(per_dataset.items()):
        dataset_instance_keys = [k for k in per_instance if k[0] == dataset]
        rules_per_instance = [per_instance[k].count_rules for k in dataset_instance_keys]
        literals_per_instance = [per_instance[k].total_rule_length for k in dataset_instance_keys]
        mean_rules_per_instance = sum(rules_per_instance) / len(rules_per_instance) if rules_per_instance else 0.0
        mean_literals_per_instance = sum(literals_per_instance) / len(literals_per_instance) if literals_per_instance else 0.0
        model_cases = sorted({k[1] for k in per_model if k[0] == dataset})

        per_dataset_rows.append(
            {
                "dataset": dataset,
                "n_model_cases": len(model_cases),
                "n_rules_total": stats.count,
                "n_instances_total": len(dataset_instance_keys),
                "mean_rules_per_instance": f"{mean_rules_per_instance:.4f}",
                "mean_literals_per_instance": f"{mean_literals_per_instance:.4f}",
                "mean_rule_length": f"{stats.mean_rule_length():.4f}",
                "median_rule_length": f"{stats.median_rule_length():.4f}",
                "max_rule_length": stats.max_rule_length,
                "mean_unique_features": f"{stats.mean_unique_features():.4f}",
                "mean_leaf_support_train": f"{stats.mean_leaf_support():.6f}",
                "mean_leaf_purity": f"{stats.mean_leaf_purity():.6f}",
                "mean_score": f"{stats.mean_score():.8f}",
                "max_score": f"{stats.max_score:.8f}",
                "model_case_ids": " | ".join(model_cases),
            }
        )

    top_rule_rows: list[dict] = []
    for key, heap in sorted(top_rules.items()):
        for rank, (_, _, payload) in enumerate(sorted(heap, key=lambda x: (x[0], x[1]), reverse=True), start=1):
            row = dict(payload)
            row["rank_within_model_case"] = rank
            top_rule_rows.append(row)

    write_csv(
        OUTPUT_DIR / "supporting_rules_summary_by_model.csv",
        per_model_rows,
        [
            "dataset",
            "kind",
            "guiding_bb",
            "percentile",
            "privacy_mode",
            "query_epsilon",
            "query_noise_mech",
            "noise_on_labeling",
            "model_name",
            "model_case_id",
            "n_rules_total",
            "n_instances",
            "mean_rules_per_instance",
            "mean_literals_per_instance",
            "mean_rule_length",
            "median_rule_length",
            "max_rule_length",
            "mean_unique_features",
            "mean_leaf_support_train",
            "mean_leaf_purity",
            "mean_score",
            "max_score",
        ],
    )

    write_csv(
        OUTPUT_DIR / "supporting_rules_summary_by_dataset.csv",
        per_dataset_rows,
        [
            "dataset",
            "n_model_cases",
            "n_rules_total",
            "n_instances_total",
            "mean_rules_per_instance",
            "mean_literals_per_instance",
            "mean_rule_length",
            "median_rule_length",
            "max_rule_length",
            "mean_unique_features",
            "mean_leaf_support_train",
            "mean_leaf_purity",
            "mean_score",
            "max_score",
            "model_case_ids",
        ],
    )

    write_csv(
        OUTPUT_DIR / "supporting_rules_top_rules.csv",
        top_rule_rows,
        [
            "dataset",
            "kind",
            "guiding_bb",
            "percentile",
            "privacy_mode",
            "query_epsilon",
            "query_noise_mech",
            "noise_on_labeling",
            "model_case_id",
            "rank_within_model_case",
            "row_id",
            "tree_idx",
            "leaf_idx",
            "rule_length",
            "unique_features",
            "leaf_support_train",
            "leaf_purity",
            "score",
            "rendered_rule",
        ],
    )

    print(f"Wrote summaries to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
