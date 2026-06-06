#!/usr/bin/env python3
"""
Summarize every per-setting rule-selection CSV stored under:
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
OUT_DIR = ROOT / "Reports-eval" / "supporting-rule-selection-tree-summary"
TOP_K = 10


def safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def safe_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def find_rule_files() -> list[tuple[str, Path]]:
    files = []
    for name in ("all_supporting_rules.csv", "selected_supporting_rules.csv"):
        for p in BASE_DIR.rglob(name):
            # skip the giant root-level aggregate dumps; keep only dataset/model_case files
            rel = p.relative_to(BASE_DIR)
            if len(rel.parts) >= 3:
                files.append(("all" if name.startswith("all_") else "selected", p))
    return sorted(files, key=lambda x: str(x[1]))


def summarize_one_file(file_kind: str, csv_path: Path):
    per_instance: dict[str, InstanceAccumulator] = defaultdict(InstanceAccumulator)
    stats = RunningStats()
    meta = None
    top_rules: list[tuple[float, int, dict[str, str]]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if meta is None:
                meta = {
                    "dataset": row["dataset"],
                    "kind": row["kind"],
                    "guiding_bb": row["guiding_bb"],
                    "percentile": row["percentile"],
                    "privacy_mode": row["privacy_mode"],
                    "query_epsilon": row["query_epsilon"],
                    "query_noise_mech": row["query_noise_mech"],
                    "noise_on_labeling": row["noise_on_labeling"],
                    "model_name": row["model_name"],
                    "model_case_id": row["model_case_id"],
                }

            rule_length = safe_int(row["rule_length"])
            unique_features = safe_int(row["unique_features"])
            leaf_support = safe_float(row["leaf_support_train"])
            leaf_purity = safe_float(row["leaf_purity"])
            score = safe_float(row["score"])
            row_id = row["row_id"]

            stats.add(rule_length, unique_features, leaf_support, leaf_purity, score)
            per_instance[row_id].add(rule_length)

            payload = {
                "dataset": row["dataset"],
                "file_kind": file_kind,
                "kind": row["kind"],
                "guiding_bb": row["guiding_bb"],
                "percentile": row["percentile"],
                "privacy_mode": row["privacy_mode"],
                "query_epsilon": row["query_epsilon"],
                "query_noise_mech": row["query_noise_mech"],
                "noise_on_labeling": row["noise_on_labeling"],
                "model_case_id": row["model_case_id"],
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
            item = (score, idx, payload)
            if len(top_rules) < TOP_K:
                heapq.heappush(top_rules, item)
            elif score > top_rules[0][0]:
                heapq.heapreplace(top_rules, item)

    if meta is None:
        return None, []

    rules_per_instance = [x.count_rules for x in per_instance.values()]
    literals_per_instance = [x.total_rule_length for x in per_instance.values()]
    meta.update(
        {
            "file_kind": file_kind,
            "csv_path": str(csv_path.relative_to(ROOT)),
            "n_rules_total": stats.count,
            "n_instances": len(per_instance),
            "mean_rules_per_instance": f"{(sum(rules_per_instance) / len(rules_per_instance)) if rules_per_instance else 0.0:.4f}",
            "mean_literals_per_instance": f"{(sum(literals_per_instance) / len(literals_per_instance)) if literals_per_instance else 0.0:.4f}",
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

    top_rule_rows = []
    for rank, (_, _, payload) in enumerate(sorted(top_rules, key=lambda x: (x[0], x[1]), reverse=True), start=1):
        payload = dict(payload)
        payload["rank_within_file"] = rank
        payload["csv_path"] = str(csv_path.relative_to(ROOT))
        top_rule_rows.append(payload)

    return meta, top_rule_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file_dir = OUT_DIR / "per_file"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    files = find_rule_files()
    if not files:
        raise FileNotFoundError(f"No per-setting rule files found under {BASE_DIR}")

    all_rows: list[dict] = []
    all_top_rules: list[dict] = []
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_kind_privacy: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for file_kind, csv_path in files:
        summary_row, top_rows = summarize_one_file(file_kind, csv_path)
        if summary_row is None:
            continue
        all_rows.append(summary_row)
        all_top_rules.extend(top_rows)
        by_dataset[summary_row["dataset"]].append(summary_row)
        key = (summary_row["file_kind"], summary_row["kind"], summary_row["privacy_mode"])
        by_kind_privacy[key].append(summary_row)

        safe_name = (
            f"{summary_row['dataset']}__{summary_row['model_case_id']}__{summary_row['file_kind']}__summary.csv"
            .replace("/", "_")
        )
        write_csv(
            per_file_dir / safe_name,
            [summary_row],
            list(summary_row.keys()),
        )

    dataset_rows = []
    for dataset, rows in sorted(by_dataset.items()):
        dataset_rows.append(
            {
                "dataset": dataset,
                "n_files": len(rows),
                "file_kinds": " | ".join(sorted({r["file_kind"] for r in rows})),
                "model_case_ids": " | ".join(sorted({r["model_case_id"] for r in rows})),
                "mean_rules_per_instance_avg": f"{sum(float(r['mean_rules_per_instance']) for r in rows) / len(rows):.4f}",
                "mean_literals_per_instance_avg": f"{sum(float(r['mean_literals_per_instance']) for r in rows) / len(rows):.4f}",
                "mean_rule_length_avg": f"{sum(float(r['mean_rule_length']) for r in rows) / len(rows):.4f}",
                "mean_score_avg": f"{sum(float(r['mean_score']) for r in rows) / len(rows):.8f}",
                "max_score_over_files": f"{max(float(r['max_score']) for r in rows):.8f}",
            }
        )

    kind_privacy_rows = []
    for (file_kind, kind, privacy_mode), rows in sorted(by_kind_privacy.items()):
        kind_privacy_rows.append(
            {
                "file_kind": file_kind,
                "kind": kind,
                "privacy_mode": privacy_mode,
                "n_files": len(rows),
                "datasets": " | ".join(sorted({r["dataset"] for r in rows})),
                "mean_rules_per_instance_avg": f"{sum(float(r['mean_rules_per_instance']) for r in rows) / len(rows):.4f}",
                "mean_literals_per_instance_avg": f"{sum(float(r['mean_literals_per_instance']) for r in rows) / len(rows):.4f}",
                "mean_rule_length_avg": f"{sum(float(r['mean_rule_length']) for r in rows) / len(rows):.4f}",
                "mean_score_avg": f"{sum(float(r['mean_score']) for r in rows) / len(rows):.8f}",
                "max_score_over_files": f"{max(float(r['max_score']) for r in rows):.8f}",
            }
        )

    write_csv(OUT_DIR / "all_files_summary.csv", all_rows, list(all_rows[0].keys()))
    write_csv(OUT_DIR / "summary_by_dataset.csv", dataset_rows, list(dataset_rows[0].keys()))
    write_csv(OUT_DIR / "summary_by_kind_privacy.csv", kind_privacy_rows, list(kind_privacy_rows[0].keys()))
    write_csv(
        OUT_DIR / "top_rules_all.csv",
        all_top_rules,
        [
            "dataset",
            "file_kind",
            "kind",
            "guiding_bb",
            "percentile",
            "privacy_mode",
            "query_epsilon",
            "query_noise_mech",
            "noise_on_labeling",
            "model_case_id",
            "rank_within_file",
            "row_id",
            "tree_idx",
            "leaf_idx",
            "rule_length",
            "unique_features",
            "leaf_support_train",
            "leaf_purity",
            "score",
            "rendered_rule",
            "csv_path",
        ],
    )

    print(f"Processed {len(all_rows)} files")
    print(f"Wrote summaries to: {OUT_DIR}")


if __name__ == "__main__":
    main()
