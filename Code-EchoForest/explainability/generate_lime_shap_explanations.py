#!/usr/bin/env python3
"""
Generate minimal LIME and SHAP explanations on the released PREMS RF surrogates.

"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from shared.project_paths import DATA_ORIGINAL_DIR, MODEL_SYNTHETIC_WISE_DIR, REPORTS_EVAL_DIR


# Balanced minimal benchmark:
# - binary: adult, activity, pol, spotify-r
# - multiclass: landsat-multi, wave-multi
DATASETS = ["adult", "activity", "pol", "spotify-r", "landsat-multi", "wave-multi"]
KIND = "logit"
GUIDING_BB = "nn"
PERCENTILE = 25

DIR_ORIGINAL = DATA_ORIGINAL_DIR
DIR_MODELS = MODEL_SYNTHETIC_WISE_DIR
DIR_REPORTS = REPORTS_EVAL_DIR / "posthoc-explainers"

RANDOM_STATE = 42
MAX_INSTANCES = 200
TOP_K_FEATURES = 10
LIME_NUM_SAMPLES = 5000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model(dataset: str):
    model_path = DIR_MODELS / dataset / f"rf_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}_wise.sav"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, model_path


def load_scaled_data(dataset: str):
    base = DIR_ORIGINAL / dataset
    train_df = pd.read_csv(base / f"train_set_{dataset}.csv")
    test_df = pd.read_csv(base / f"test_set_{dataset}.csv")

    train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.startswith("Unnamed")]

    feature_names = train_df.columns.tolist()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df.values.astype(np.float32))
    x_test = scaler.transform(test_df.values.astype(np.float32))

    return (
        x_train.astype(np.float32),
        x_test.astype(np.float32),
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        feature_names,
    )


def sample_test_indices(n_rows: int) -> np.ndarray:
    if n_rows <= MAX_INSTANCES:
        return np.arange(n_rows)
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n_rows, size=MAX_INSTANCES, replace=False))


def compute_lime_rows(model, x_train, x_test, feature_names, selected_idx, dataset):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "LIME not installed. Please install `lime` on the server environment."
        ) from e

    class_names = [str(c) for c in getattr(model, "classes_", [])]
    explainer = LimeTabularExplainer(
        training_data=x_train,
        feature_names=feature_names,
        class_names=class_names if class_names else None,
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE,
    )

    rows = []
    examples = []
    t0 = time.perf_counter()
    for local_rank, idx in enumerate(selected_idx):
        x = x_test[idx]
        pred_class = int(model.predict(x.reshape(1, -1))[0])
        exp = explainer.explain_instance(
            x,
            model.predict_proba,
            num_features=min(TOP_K_FEATURES, x.shape[0]),
            num_samples=LIME_NUM_SAMPLES,
            top_labels=1,
        )
        mapping = exp.as_map()
        local = mapping.get(pred_class)
        if local is None and mapping:
            local = next(iter(mapping.values()))
        local = local or []

        ranked = sorted(local, key=lambda z: abs(z[1]), reverse=True)
        if local_rank < 3:
            example_lines = [f"instance={idx} pred={pred_class}"]
            for rank, (feat_idx, weight) in enumerate(ranked[:TOP_K_FEATURES], start=1):
                example_lines.append(
                    f"  {rank}. {feature_names[int(feat_idx)]} -> {weight:.6f}"
                )
            examples.append("\n".join(example_lines))

        for rank, (feat_idx, weight) in enumerate(ranked[:TOP_K_FEATURES], start=1):
            feat_idx = int(feat_idx)
            rows.append(
                {
                    "dataset": dataset,
                    "explainer": "lime",
                    "instance_id": int(idx),
                    "pred_class": pred_class,
                    "rank": rank,
                    "feature_idx": feat_idx,
                    "feature_name": feature_names[feat_idx],
                    "feature_value_scaled": float(x[feat_idx]),
                    "importance": float(weight),
                    "abs_importance": float(abs(weight)),
                }
            )
    elapsed = time.perf_counter() - t0
    return rows, examples, elapsed


def _normalize_shap_values(shap_values, pred_indices):
    """
    Return per-instance per-feature SHAP values for the predicted class.
    Compatible with common SHAP outputs for tree classifiers.
    """
    sv = shap_values
    if isinstance(sv, list):
        # old multiclass API: list[n_classes] of (n_samples, n_features)
        out = np.stack([sv[pred_indices[i]][i] for i in range(len(pred_indices))], axis=0)
        return out

    sv = np.asarray(sv)
    if sv.ndim == 2:
        # binary old API or regression
        return sv
    if sv.ndim == 3:
        # either (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
        if sv.shape[2] == len(np.unique(pred_indices)) or sv.shape[2] > 1:
            # assume (n_samples, n_features, n_classes)
            return np.stack([sv[i, :, pred_indices[i]] for i in range(len(pred_indices))], axis=0)
        return np.stack([sv[i, pred_indices[i], :] for i in range(len(pred_indices))], axis=0)
    raise ValueError(f"Unsupported SHAP value shape: {sv.shape}")


def compute_shap_rows(model, x_train, x_test, feature_names, selected_idx, dataset):
    try:
        import shap
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "SHAP not installed. Please install `shap` on the server environment."
        ) from e

    x_sel = x_test[selected_idx]
    pred = model.predict(x_sel)
    pred_classes = np.array([int(c) for c in pred], dtype=int)

    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sel, check_additivity=False)
    shap_pred = _normalize_shap_values(shap_values, pred_classes)
    elapsed = time.perf_counter() - t0

    rows = []
    examples = []
    for local_rank, idx in enumerate(selected_idx):
        contrib = shap_pred[local_rank]
        abs_order = np.argsort(np.abs(contrib))[::-1][: min(TOP_K_FEATURES, len(contrib))]

        if local_rank < 3:
            example_lines = [f"instance={idx} pred={pred_classes[local_rank]}"]
            for rank, feat_idx in enumerate(abs_order, start=1):
                example_lines.append(
                    f"  {rank}. {feature_names[int(feat_idx)]} -> {float(contrib[feat_idx]):.6f}"
                )
            examples.append("\n".join(example_lines))

        for rank, feat_idx in enumerate(abs_order, start=1):
            feat_idx = int(feat_idx)
            val = float(contrib[feat_idx])
            rows.append(
                {
                    "dataset": dataset,
                    "explainer": "shap",
                    "instance_id": int(idx),
                    "pred_class": int(pred_classes[local_rank]),
                    "rank": rank,
                    "feature_idx": feat_idx,
                    "feature_name": feature_names[feat_idx],
                    "feature_value_scaled": float(x_sel[local_rank, feat_idx]),
                    "importance": val,
                    "abs_importance": abs(val),
                }
            )
    return rows, examples, elapsed


def summarize(rows: pd.DataFrame, runtimes: list[dict]) -> pd.DataFrame:
    grouped = (
        rows.groupby(["dataset", "explainer"])
        .agg(
            n_instances=("instance_id", "nunique"),
            mean_abs_importance=("abs_importance", "mean"),
            median_abs_importance=("abs_importance", "median"),
            max_abs_importance=("abs_importance", "max"),
            mean_signed_importance=("importance", "mean"),
            top1_mean_abs_importance=("abs_importance", lambda s: s[rows.loc[s.index, "rank"] == 1].mean()),
        )
        .reset_index()
    )
    runtime_df = pd.DataFrame(runtimes)
    return grouped.merge(runtime_df, on=["dataset", "explainer"], how="left")


def main():
    ensure_dir(DIR_REPORTS)
    all_rows = []
    runtime_rows = []

    for dataset in DATASETS:
        model, model_path = load_model(dataset)
        x_train, x_test, _, _, feature_names = load_scaled_data(dataset)
        selected_idx = sample_test_indices(len(x_test))

        lime_rows, lime_examples, lime_time = compute_lime_rows(
            model, x_train, x_test, feature_names, selected_idx, dataset
        )
        shap_rows, shap_examples, shap_time = compute_shap_rows(
            model, x_train, x_test, feature_names, selected_idx, dataset
        )

        all_rows.extend(lime_rows)
        all_rows.extend(shap_rows)

        runtime_rows.append(
            {
                "dataset": dataset,
                "explainer": "lime",
                "model_path": str(model_path),
                "n_instances": len(selected_idx),
                "top_k_features": TOP_K_FEATURES,
                "elapsed_seconds": round(lime_time, 4),
                "elapsed_per_instance_seconds": round(lime_time / max(len(selected_idx), 1), 6),
            }
        )
        runtime_rows.append(
            {
                "dataset": dataset,
                "explainer": "shap",
                "model_path": str(model_path),
                "n_instances": len(selected_idx),
                "top_k_features": TOP_K_FEATURES,
                "elapsed_seconds": round(shap_time, 4),
                "elapsed_per_instance_seconds": round(shap_time / max(len(selected_idx), 1), 6),
            }
        )

        example_path = DIR_REPORTS / f"lime_shap_examples_{dataset}.txt"
        with example_path.open("w", encoding="utf-8") as f:
            f.write(f"DATASET: {dataset}\n")
            f.write("MODEL: released PREMS RF surrogate\n\n")
            f.write("[LIME]\n")
            f.write("\n\n".join(lime_examples))
            f.write("\n\n[SHAP]\n")
            f.write("\n\n".join(shap_examples))

    df = pd.DataFrame(all_rows)
    df.to_csv(DIR_REPORTS / "lime_shap_explanations.csv", index=False)

    summary_df = summarize(df, runtime_rows)
    summary_df.to_csv(DIR_REPORTS / "lime_shap_summary_by_dataset.csv", index=False)

    print(f"[OK] wrote {DIR_REPORTS / 'lime_shap_explanations.csv'}")
    print(f"[OK] wrote {DIR_REPORTS / 'lime_shap_summary_by_dataset.csv'}")


if __name__ == "__main__":
    main()
