"""
Summarize structural properties of trained EchoForest surrogates.

"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROOT = Path("../Model-synthetic-wise")
OUT_DIR = Path("../Reports-eval") / "validation"
OUT_CSV = OUT_DIR / "echo_forest_structure_summary.csv"


def parse_model_name(path: Path) -> dict:
    name = path.stem
    patterns = [
        re.compile(
            r"^rf_(?P<dataset>.+)_(?P<kind>entropy|margin|kappa|logit)_(?P<bb>rf|nn)_(?P<percentile>\d+)_wise$"
        ),
        re.compile(
            r"^rf_dpquery_(?P<dataset>.+)_(?P<kind>entropy|margin|kappa|logit)_(?P<bb>rf|nn)_(?P<percentile>\d+).*$"
        ),
        re.compile(
            r"^rf_dp_(?P<epsilon>\d+(?:\.\d+)?)_(?P<mech>laplace|gaussian)_(?P<dataset>.+)_(?P<kind>entropy|margin|kappa|logit)_(?P<bb>rf|nn)_(?P<percentile>\d+)$"
        ),
        re.compile(r"^rf_(?P<dataset>.+)_wise$"),
    ]
    for pattern in patterns:
        match = pattern.match(name)
        if match:
            meta = match.groupdict()
            meta["model_name"] = path.name
            return meta
    return {"dataset": path.parent.name, "model_name": path.name}


def summarize_forest(model_path: Path) -> dict:
    with open(model_path, "rb") as handle:
        rf = pickle.load(handle)

    estimators = getattr(rf, "estimators_", [])
    depths = [tree.tree_.max_depth for tree in estimators]
    node_counts = [tree.tree_.node_count for tree in estimators]
    leaf_counts = [tree.tree_.n_leaves for tree in estimators]

    meta = parse_model_name(model_path)
    meta.update(
        {
            "dataset_dir": model_path.parent.name,
            "n_trees": len(estimators),
            "configured_max_depth": getattr(rf, "max_depth", None),
            "mean_tree_depth": float(np.mean(depths)) if depths else np.nan,
            "median_tree_depth": float(np.median(depths)) if depths else np.nan,
            "max_tree_depth": int(np.max(depths)) if depths else np.nan,
            "mean_node_count": float(np.mean(node_counts)) if node_counts else np.nan,
            "mean_leaf_count": float(np.mean(leaf_counts)) if leaf_counts else np.nan,
            "model_path": str(model_path),
        }
    )
    return meta


def collect_model_paths() -> list[Path]:
    return sorted(MODEL_ROOT.glob("*/*.sav"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_forest(path) for path in collect_model_paths()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
