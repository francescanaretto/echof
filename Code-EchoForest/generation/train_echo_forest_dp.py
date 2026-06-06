#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pickle
from joblib import load as joblib_load, dump as joblib_dump

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import f1_score

import torch
from diffprivlib.models import RandomForestClassifier as DPRandomForestClassifier
from loaders import load_trained_nn

#           Configuration
DATASETS   = ["spotify"]

KIND       = "entropy"
GUIDING_BB = "nn"
PERCENTILE = 50

DIR_SYNTH        = Path("../Data-synthetic/wise")
DIR_MODELS_BB    = Path("../Model-original")
DIR_MODELS_STUD  = Path("../Model-synthetic-wise")

TEST_SIZE_SYNTH  = 0.30
RANDOM_STATE     = 42

DP_EPSILON = 1  # fixed

# Shared grid used for both standard and DP EchoForest training.
# We keep only the hyperparameters available in both implementations.
DP_RF_PARAM_GRID = {
    "n_estimators": [50, 70, 100, 150, 200],
    "max_depth":    [None, 5, 10, 15, 20],
}
# Utilities
def load_wise_synthetic(dataset, kind, guiding_bb, percentile):
    path = DIR_SYNTH / dataset / f"synthetic_19_checks_{dataset}_{kind}_{guiding_bb}_{percentile}.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    return X, y


def load_original_train_test_raw(dataset):
    base = Path("../Data-original") / dataset
    Xtr = pd.read_csv(base / f"train_set_{dataset}.csv")
    Xte = pd.read_csv(base / f"test_set_{dataset}.csv")
    Xtr = Xtr.loc[:, ~Xtr.columns.str.startswith("Unnamed")]
    Xte = Xte.loc[:, ~Xte.columns.str.startswith("Unnamed")]
    return Xtr.values.astype(np.float32), Xte.values.astype(np.float32)


def try_load_nn_scaler(dataset):
    cand = [
        DIR_MODELS_BB / dataset / f"scaler_{dataset}.joblib",
        DIR_MODELS_BB / dataset / "scaler.joblib",
    ]
    for p in cand:
        if p.exists():
            return joblib_load(p)
    return None


@torch.no_grad()
def predict_nn_in_batches(model, X, batch_size=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xt = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
        logits = model(xt)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        out.append(pred)
    return np.concatenate(out)


def label_original_with_bb(dataset, Xtr_raw, Xte_raw):
    model = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS_BB))
    scaler = try_load_nn_scaler(dataset)

    if scaler is None:
        scaler = StandardScaler().fit(Xtr_raw)

    Xtr = scaler.transform(Xtr_raw).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)

    ytr = predict_nn_in_batches(model, Xtr)
    yte = predict_nn_in_batches(model, Xte)

    return Xtr, Xte, ytr, yte


def compute_bounds(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    maxs[mins == maxs] += 1e-6
    return (mins, maxs)


def save_dp_rf_model(model, dataset, kind, guiding_bb, percentile):
    out_dir = DIR_MODELS_STUD / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"rf_dp{DP_EPSILON}_{dataset}_{kind}_{guiding_bb}_{percentile}_wise.sav"

    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] saved: {out_path}")


# Model selection
def select_dp_rf(X_synth, y_synth, X_eval, y_eval):
    grid = list(ParameterGrid(DP_RF_PARAM_GRID))

    bounds = compute_bounds(X_synth)
    classs = np.unique(y_synth)

    best_model = None
    best_score = -1
    best_params = None

    for params in grid:
        print(f"\n[config] {params}")

        model = DPRandomForestClassifier(
            epsilon=DP_EPSILON,
            bounds=bounds,
            classs=classs,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            shuffle=True,
            random_state=RANDOM_STATE,
            n_jobs=20
        )

        model.fit(X_synth, y_synth)

        pred = model.predict(X_eval)
        score = f1_score(y_eval, pred, average="macro")

        print(f" -> F1_macro: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
            print(" [best] updated")

    return best_model, best_params, best_score


# Main
def main():
    for dataset in DATASETS:
        print(f"\n=== {dataset} | DP-RF ε={DP_EPSILON} ===")

        X_synth, y_synth = load_wise_synthetic(dataset, KIND, GUIDING_BB, PERCENTILE)
        Xtr_raw, Xte_raw = load_original_train_test_raw(dataset)

        Xtr, Xte, ytr_bb, yte_bb = label_original_with_bb(dataset, Xtr_raw, Xte_raw)

        X_eval = np.vstack([Xtr, Xte])
        y_eval = np.concatenate([ytr_bb, yte_bb])

        best_model, best_params, best_score = select_dp_rf(
            X_synth, y_synth, X_eval, y_eval
        )

        print("\nBest params:", best_params)
        print("Best F1:", best_score)

        save_dp_rf_model(best_model, dataset, KIND, GUIDING_BB, PERCENTILE)


if __name__ == "__main__":
    main()
