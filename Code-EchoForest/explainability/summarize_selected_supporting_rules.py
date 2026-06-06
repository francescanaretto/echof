#!/usr/bin/env python3
"""
Create compact summaries from selected_supporting_rules.csv and, when available,
compare them against all_supporting_rules.csv.

"""

from __future__ import annotations

import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "Reports-eval" / "supporting-rule-selection"
INPUT_SELECTED = BASE_DIR / "selected_supporting_rules.csv"
INPUT_ALL = BASE_DIR / "all_supporting_rules.csv"
OUTPUT_DIR = ROOT / "Reports-eval" / "supporting-rule-selection-summary-selected"
TOP_K = 20


def safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def safe_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def pct_reduction(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return 100.0 * (before - after) / before


def ratio(after: float, before: float) -> float:
    if before == 0:
        return 0.0
    return after / before


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

    def add(self, rule_length: int) -> None:
        self.count_rules += 1
        self.total_rule_length += rule_length


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_csv(csv_path: Path, top_k: int):
    per_model: dict[tuple[str, str], RunningStats] = {}
    per_dataset: dict[str, RunningStats] = {}
    per_instance: dict[tuple[str, str, str], InstanceAccumulator] = defaultdict(InstanceAccumulator)
    model_meta: dict[tuple[str, str], dict[str, str]] = {}
    top_rules: dict[tuple[str, str], list[tuple[float, int, dict[str, str]]]] = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
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
            per_instance[(dataset, model_case_id, row_id)].add(rule_length)

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
                "row_id": row["row_id"],
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
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    return per_model, per_dataset, per_instance, model_meta, top_rules


def build_summary_rows(per_model, per_dataset, per_instance, model_meta):
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
    return per_model_rows, per_dataset_rows


def build_top_rule_rows(top_rules):
    rows: list[dict] = []
    for _, heap in sorted(top_rules.items()):
        for rank, (_, _, payload) in enumerate(sorted(heap, key=lambda x: (x[0], x[1]), reverse=True), start=1):
            row = dict(payload)
            row["rank_within_model_case"] = rank
            rows.append(row)
    return rows


def comparison_rows(selected_model_rows, all_model_rows, selected_dataset_rows, all_dataset_rows):
    all_by_model = {(r["dataset"], r["model_case_id"]): r for r in all_model_rows}
    sel_by_model = {(r["dataset"], r["model_case_id"]): r for r in selected_model_rows}

    comp_model_rows = []
    for key, sel in sorted(sel_by_model.items()):
        allr = all_by_model.get(key)
        if not allr:
            continue
        comp_model_rows.append(
            {
                "dataset": sel["dataset"],
                "model_case_id": sel["model_case_id"],
                "kind": sel["kind"],
                "guiding_bb": sel["guiding_bb"],
                "percentile": sel["percentile"],
                "privacy_mode": sel["privacy_mode"],
                "selected_n_rules_total": sel["n_rules_total"],
                "all_n_rules_total": allr["n_rules_total"],
                "rules_reduction_pct": f"{pct_reduction(float(allr['n_rules_total']), float(sel['n_rules_total'])):.4f}",
                "selected_mean_rules_per_instance": sel["mean_rules_per_instance"],
                "all_mean_rules_per_instance": allr["mean_rules_per_instance"],
                "mean_rules_per_instance_reduction_pct": f"{pct_reduction(float(allr['mean_rules_per_instance']), float(sel['mean_rules_per_instance'])):.4f}",
                "selected_mean_literals_per_instance": sel["mean_literals_per_instance"],
                "all_mean_literals_per_instance": allr["mean_literals_per_instance"],
                "mean_literals_per_instance_reduction_pct": f"{pct_reduction(float(allr['mean_literals_per_instance']), float(sel['mean_literals_per_instance'])):.4f}",
                "selected_mean_rule_length": sel["mean_rule_length"],
                "all_mean_rule_length": allr["mean_rule_length"],
                "selected_mean_score": sel["mean_score"],
                "all_mean_score": allr["mean_score"],
                "score_retention_ratio": f"{ratio(float(sel['mean_score']), float(allr['mean_score'])):.6f}",
                "selected_mean_leaf_support_train": sel["mean_leaf_support_train"],
                "all_mean_leaf_support_train": allr["mean_leaf_support_train"],
                "selected_mean_leaf_purity": sel["mean_leaf_purity"],
                "all_mean_leaf_purity": allr["mean_leaf_purity"],
            }
        )

    all_by_dataset = {r["dataset"]: r for r in all_dataset_rows}
    sel_by_dataset = {r["dataset"]: r for r in selected_dataset_rows}
    comp_dataset_rows = []
    for dataset, sel in sorted(sel_by_dataset.items()):
        allr = all_by_dataset.get(dataset)
        if not allr:
            continue
        comp_dataset_rows.append(
            {
                "dataset": dataset,
                "selected_n_rules_total": sel["n_rules_total"],
                "all_n_rules_total": allr["n_rules_total"],
                "rules_reduction_pct": f"{pct_reduction(float(allr['n_rules_total']), float(sel['n_rules_total'])):.4f}",
                "selected_mean_rules_per_instance": sel["mean_rules_per_instance"],
                "all_mean_rules_per_instance": allr["mean_rules_per_instance"],
                "mean_rules_per_instance_reduction_pct": f"{pct_reduction(float(allr['mean_rules_per_instance']), float(sel['mean_rules_per_instance'])):.4f}",
                "selected_mean_literals_per_instance": sel["mean_literals_per_instance"],
                "all_mean_literals_per_instance": allr["mean_literals_per_instance"],
                "mean_literals_per_instance_reduction_pct": f"{pct_reduction(float(allr['mean_literals_per_instance']), float(sel['mean_literals_per_instance'])):.4f}",
                "selected_mean_rule_length": sel["mean_rule_length"],
                "all_mean_rule_length": allr["mean_rule_length"],
                "selected_mean_score": sel["mean_score"],
                "all_mean_score": allr["mean_score"],
                "score_retention_ratio": f"{ratio(float(sel['mean_score']), float(allr['mean_score'])):.6f}",
                "selected_mean_leaf_support_train": sel["mean_leaf_support_train"],
                "all_mean_leaf_support_train": allr["mean_leaf_support_train"],
                "selected_mean_leaf_purity": sel["mean_leaf_purity"],
                "all_mean_leaf_purity": allr["mean_leaf_purity"],
            }
        )
    return comp_model_rows, comp_dataset_rows


def main() -> None:
    if not INPUT_SELECTED.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_SELECTED}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sel = summarize_csv(INPUT_SELECTED, TOP_K)
    sel_model_rows, sel_dataset_rows = build_summary_rows(*sel[:4])
    sel_top_rule_rows = build_top_rule_rows(sel[4])

    write_csv(
        OUTPUT_DIR / "selected_rules_summary_by_model.csv",
        sel_model_rows,
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
        OUTPUT_DIR / "selected_rules_summary_by_dataset.csv",
        sel_dataset_rows,
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
        OUTPUT_DIR / "selected_rules_top_rules.csv",
        sel_top_rule_rows,
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

    if INPUT_ALL.exists():
        all_sum = summarize_csv(INPUT_ALL, top_k=1)
        all_model_rows, all_dataset_rows = build_summary_rows(*all_sum[:4])
        comp_model_rows, comp_dataset_rows = comparison_rows(
            sel_model_rows, all_model_rows, sel_dataset_rows, all_dataset_rows
        )

        write_csv(
            OUTPUT_DIR / "selected_vs_all_comparison_by_model.csv",
            comp_model_rows,
            [
                "dataset",
                "model_case_id",
                "kind",
                "guiding_bb",
                "percentile",
                "privacy_mode",
                "selected_n_rules_total",
                "all_n_rules_total",
                "rules_reduction_pct",
                "selected_mean_rules_per_instance",
                "all_mean_rules_per_instance",
                "mean_rules_per_instance_reduction_pct",
                "selected_mean_literals_per_instance",
                "all_mean_literals_per_instance",
                "mean_literals_per_instance_reduction_pct",
                "selected_mean_rule_length",
                "all_mean_rule_length",
                "selected_mean_score",
                "all_mean_score",
                "score_retention_ratio",
                "selected_mean_leaf_support_train",
                "all_mean_leaf_support_train",
                "selected_mean_leaf_purity",
                "all_mean_leaf_purity",
            ],
        )

        write_csv(
            OUTPUT_DIR / "selected_vs_all_comparison_by_dataset.csv",
            comp_dataset_rows,
            [
                "dataset",
                "selected_n_rules_total",
                "all_n_rules_total",
                "rules_reduction_pct",
                "selected_mean_rules_per_instance",
                "all_mean_rules_per_instance",
                "mean_rules_per_instance_reduction_pct",
                "selected_mean_literals_per_instance",
                "all_mean_literals_per_instance",
                "mean_literals_per_instance_reduction_pct",
                "selected_mean_rule_length",
                "all_mean_rule_length",
                "selected_mean_score",
                "all_mean_score",
                "score_retention_ratio",
                "selected_mean_leaf_support_train",
                "all_mean_leaf_support_train",
                "selected_mean_leaf_purity",
                "all_mean_leaf_purity",
            ],
        )

    print(f"Wrote summaries to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
