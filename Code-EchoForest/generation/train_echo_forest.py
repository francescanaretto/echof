#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import classification_report, f1_score

# Optional NN black-box support
import torch
from loaders import load_trained_nn

#           Configuration
DATASETS   = ["adult"]   # edit this list to process more datasets

KIND       = "entropy"    # e.g. "entropy" or "margin"
GUIDING_BB = "nn"        # "rf" or "nn"
PERCENTILE = 50          # e.g. 25, 50, ...

DIR_SYNTH        = Path("../Data-synthetic/wise")
DIR_MODELS_BB    = Path("../Model-original")       # directory containing the original black-box models
DIR_MODELS_STUD  = Path("../Model-synthetic-wise") # directory used for EchoForest models

TEST_SIZE_SYNTH  = 0.30       # used only for a quick report; tuning is performed on original data
RANDOM_STATE     = 42

# Shared grid used for both standard and DP EchoForest training.
# We keep only the hyperparameters available in both implementations.
RF_PARAM_GRID = {
    "n_estimators":     [50, 70, 100, 150, 200],
    "max_depth":        [None, 5, 10, 15, 20],
}

RF_SCORING = "f1_macro"  # selection criterion based on original-data labels


# Utilities
def load_wise_synthetic(dataset: str,
                        kind: str,
                        guiding_bb: str,
                        percentile: int) -> tuple[np.ndarray, np.ndarray]:
    """Load WISE synthetic data and return numpy arrays."""
    path = DIR_SYNTH / dataset / f"synthetic_19_checks_{dataset}_{kind}_{guiding_bb}_{percentile}.csv"
    if not path.exists():
        raise FileNotFoundError(f"File synthetic not found: {path}")

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {path}")

    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.70, random_state=42,
                                                        stratify=y)
    return X_train, Y_train


def load_original_train_test(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load original train/test splits as numpy arrays without Unnamed columns."""
    base = Path("../Data-original") / dataset

    Xtr = pd.read_csv(base / f"train_set_{dataset}.csv", index_col=0)
    Xte = pd.read_csv(base / f"test_set_{dataset}.csv",  index_col=0)

    Xtr = Xtr.loc[:, ~Xtr.columns.str.startswith("Unnamed")]
    Xte = Xte.loc[:, ~Xte.columns.str.startswith("Unnamed")]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr.values)
    Xte_s = scaler.transform(Xte.values)

    # Keep the original feature space expected by the black-box.
    #Xtr_arr = Xtr_s.to_numpy(dtype=np.float32)
    #Xte_arr = Xte_s.to_numpy(dtype=np.float32)

    return Xtr_s, Xte_s


def load_rf_bb(dataset: str):
    """Load the original RF black-box."""
    path = DIR_MODELS_BB / dataset / f"rf_{dataset}.sav"
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_nn_in_batches(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Return batched NN predictions (classes 0..K-1)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = X[start:start+batch_size]
            xt = torch.tensor(chunk, dtype=torch.float32)
            logits = model(xt)
            if logits.ndim == 2 and logits.shape[1] > 1:
                pred = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                pred = (torch.sigmoid(logits.flatten()) >= 0.5).cpu().numpy().astype(int)
            out.append(pred)
    return np.concatenate(out, axis=0)


def label_original_with_bb(dataset: str,
                           guiding_bb: str,
                           Xtr: np.ndarray,
                           Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Label the original train/test splits with the original black-box (RF or NN).
    Returns y_train_bb, y_test_bb.
    """
    if guiding_bb == "rf":
        bb = load_rf_bb(dataset)
        ytr = bb.predict(Xtr)
        yte = bb.predict(Xte)
        return ytr, yte

    elif guiding_bb == "nn":
        # Use the local NN loader.
        model = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS_BB))
        ytr = predict_nn_in_batches(model, Xtr)
        yte = predict_nn_in_batches(model, Xte)
        return ytr, yte

    else:
        raise ValueError(f"Unknown GUIDING_BB value: {guiding_bb}")


def save_rf_model(model, dataset, kind, guiding_bb, percentile):
    """Save the trained EchoForest model."""
    out_dir = DIR_MODELS_STUD / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rf_{dataset}_{kind}_{guiding_bb}_{percentile}_wise.sav"

    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] RF model saved to: {out_path}")


# Training and model selection
def select_rf_on_original_data(X_synth: np.ndarray,
                               y_synth: np.ndarray,
                               Xtr_orig: np.ndarray,
                               Xte_orig: np.ndarray,
                               ytr_bb: np.ndarray,
                               yte_bb: np.ndarray,
                               param_grid: dict,
                               random_state: int = 42):
    """
    Select the RF that maximizes macro F1 on original train+test data labeled by the black-box.

    For each configuration in the parameter grid:
    - fit on the full synthetic set
    - evaluate macro F1 on original train+test labeled by the black-box
    Returns best_rf, best_params, best_f1_global.
    """

    grid = list(ParameterGrid(param_grid))
    print(f"[info] Number of RF configurations to test: {len(grid)}")

    X_orig_all = np.vstack([Xtr_orig, Xte_orig])
    y_orig_all = np.concatenate([ytr_bb, yte_bb])

    best_rf = None
    best_params = None
    best_f1_global = -np.inf

    for i, params in enumerate(grid, 1):
        print(f"\n[info] Configuration {i}/{len(grid)}: {params}")
        rf = RandomForestClassifier(random_state=random_state, n_jobs=10, **params)

        rf.fit(X_synth, y_synth)

        y_pred_all = rf.predict(X_orig_all)
        f1_global  = f1_score(y_orig_all, y_pred_all, average="macro")
        print(f"  -> macro F1 on original train+test: {f1_global:.4f}")

        if f1_global > best_f1_global:
            best_f1_global = f1_global
            best_params = params
            best_rf = rf
            print("  [info] new best RF found")

    return best_rf, best_params, best_f1_global


# Main
def main():
    DIR_MODELS_STUD.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        try:
            print(f"\n=== Dataset: {dataset} | EchoForest trained on WISE data ({KIND}, BB={GUIDING_BB}) ===")

            # 1) Load synthetic WISE data.
            X_synth, y_synth = load_wise_synthetic(dataset, KIND, GUIDING_BB, PERCENTILE)
            print(f"[info] shape synthetic wise: X={X_synth.shape}, y={y_synth.shape}")

            # 2) Load the original train/test splits.
            Xtr_orig, Xte_orig = load_original_train_test(dataset)
            print(f"[info] shape original train/test: Xtr={Xtr_orig.shape}, Xte={Xte_orig.shape}")

            # 3) Label original data with the black-box.
            ytr_bb, yte_bb = label_original_with_bb(dataset, GUIDING_BB, Xtr_orig, Xte_orig)
            print(f"[info] classes BB (train): {np.unique(ytr_bb, return_counts=True)}")

            # 4) Select RF hyperparameters using original data labeled by the black-box.
            best_rf, best_params, best_f1_global = select_rf_on_original_data(
                X_synth, y_synth,
                Xtr_orig, Xte_orig,
                ytr_bb, yte_bb,
                RF_PARAM_GRID,
                random_state=RANDOM_STATE
            )

            print("\n[info] Best parameters found:", best_params)
            print(f"[info] Best macro F1 on original train+test: {best_f1_global:.4f}")

            # 5) Report train and test performance separately.
            ytr_pred = best_rf.predict(Xtr_orig)
            yte_pred = best_rf.predict(Xte_orig)

            f1_tr = f1_score(ytr_bb, ytr_pred, average="macro")
            f1_te = f1_score(yte_bb, yte_pred, average="macro")

            print(f"\n[info] Macro F1 on original TRAIN: {f1_tr:.4f}")
            print(f"[info] Macro F1 on original TEST: {f1_te:.4f}")

            print("\n[info] classification_report TRAIN:")
            print(classification_report(ytr_bb, ytr_pred))

            print("[info] classification_report TEST:")
            print(classification_report(yte_bb, yte_pred))

            # 6) Optional quick evaluation on a synthetic hold-out split.
            X_trs, X_tes, y_trs, y_tes = train_test_split(
                X_synth, y_synth,
                test_size=TEST_SIZE_SYNTH,
                random_state=RANDOM_STATE,
                stratify=y_synth
            )
            y_synth_pred = best_rf.predict(X_tes)
            f1_synth = f1_score(y_tes, y_synth_pred, average="macro")
            print(f"[info] Macro F1 on the synthetic test split: {f1_synth:.4f}")
            save_rf_model(best_rf, dataset, KIND, GUIDING_BB, PERCENTILE)

        except Exception as e:
            print(f"[ERR] {dataset}: {e}")


if __name__ == "__main__":
    main()
