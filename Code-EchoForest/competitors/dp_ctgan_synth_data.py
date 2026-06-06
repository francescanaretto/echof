#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Limit BLAS/OMP threads before importing numpy/torch.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import typing

try:
    from typing_extensions import Self as _Self
    typing.Self = _Self  # type: ignore[attr-defined]
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
if not hasattr(nn, "RMSNorm"):
    class _RMSNormShim(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            self._ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        def forward(self, x):
            return self._ln(x)
    nn.RMSNorm = _RMSNormShim
# synthcity: DPGAN
from synthcity.plugins import Plugins

# If these helpers already exist in loaders.py, import them instead of
# redefining them here.
from loaders import load_trained_nn


# ============ CONFIG ============
DATASETS   = ["adult", "activity", "pol", "spotify", "spotify-r", "landsat", "landsat2", "electricity"]
DIR_DATA   = Path("../Data-original")
DIR_MODELS = Path("../Model-original")
DIR_SYNTH  = Path("../Data-synthetic")

# DP-GAN generation
DP_EPSILON      = 1.0
DP_DELTA        = 1e-5
N_ITERS         = 5000
N_SYNTH         = 80000          # or "match" to use len(train_set)
MAX_SYNTH_CAP   = 80000

TEST_SIZE       = 0.30
RANDOM_STATE    = 42
BATCH_SIZE      = 4096


# ============ UTILITIES ============

def load_train_df(dataset: str) -> pd.DataFrame:
    """Load train_set_{dataset}.csv and drop any 'Unnamed:*' columns."""
    p = DIR_DATA / dataset / f"train_set_{dataset}.csv"
    df = pd.read_csv(p, index_col=0)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


def save_split(dataset: str, kind: str, bb_tag: str, Xtr, Xte, Ytr, Yte):
    """Save the synthetic train/test splits."""
    outdir = DIR_SYNTH / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{dataset}_{kind}_{bb_tag}"

    pd.DataFrame(Xtr).to_csv(outdir / f"train_set_synth_{base}.csv", index=False)
    pd.DataFrame(Xte).to_csv(outdir / f"test_set_synth_{base}.csv", index=False)
    pd.Series(Ytr, name="label").to_csv(outdir / f"train_labels_synth_{base}.csv", index=False)
    pd.Series(Yte, name="label").to_csv(outdir / f"test_labels_synth_{base}.csv", index=False)


def load_rf_model(dataset: str):
    """Load the Random Forest trained on real data."""
    with open(DIR_MODELS / dataset / f"rf_{dataset}.sav", "rb") as f:
        return pickle.load(f)


def align_to_model_feature_names(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align X to the feature names and ordering used when fitting the RF.
    Handles missing columns such as 'Unnamed: 0'.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = list(X.columns)

    X = X.copy()
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        if c.startswith("Unnamed"):
            # Recreate a deterministic proxy column.
            if np.issubdtype(X.index.dtype, np.number):
                X[c] = X.index.astype(float)
            else:
                X[c] = np.arange(len(X), dtype=float)
        else:
            X[c] = 0
    return X.reindex(columns=expected, fill_value=0)


# ---- utilities for NN (PyTorch) ----

def get_expected_in_features(model: nn.Module) -> int:
    """Return the input dimensionality of the first Linear layer."""
    first_linear = None
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        for m in model.net:
            if isinstance(m, nn.Linear):
                first_linear = m
                break
    if first_linear is None and hasattr(model, "lin1"):
        first_linear = model.lin1
    if first_linear is None:
        raise RuntimeError("Could not find the first Linear layer in the network.")
    return int(first_linear.in_features)


def build_X_for_nn(Xref_real: pd.DataFrame,
                   X_synth: pd.DataFrame,
                   expected_in: int) -> pd.DataFrame:
    """
    Build a numeric float32 DataFrame for the NN:
    - use only real-data numeric columns also available in the synthetic data
    - impute missing values with the real-data median
    - keep the first expected_in columns
    """
    num_cols_real = Xref_real.select_dtypes(include=[np.number]).columns.tolist()
    common = [c for c in num_cols_real if c in X_synth.columns]
    if len(common) < expected_in:
        raise RuntimeError(
            f"Not enough shared numeric features ({len(common)}) for an NN expecting "
            f"{expected_in} inputs."
        )

    X = X_synth[common].copy()
    for c in common:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = pd.to_numeric(Xref_real[c], errors="coerce").median()
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med).astype(np.float32)

    return X.iloc[:, :expected_in]


def predict_nn_in_batches(model: nn.Module,
                          X_df: pd.DataFrame,
                          batch_size: int = 4096) -> np.ndarray:
    """Run batched NN predictions and return a numpy label vector."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_df), batch_size):
            chunk = X_df.iloc[start:start + batch_size]
            xt = torch.tensor(chunk.to_numpy(dtype=np.float32), dtype=torch.float32)
            logits = model(xt)
            if logits.ndim == 2 and logits.shape[1] > 1:
                pred = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                pred = (torch.sigmoid(logits.flatten()) >= 0.5).cpu().numpy().astype(int)
            out.append(pred)
    return np.concatenate(out, axis=0)


# ============ MAIN LOOP ============

def main():
    DIR_SYNTH.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        try:
            print(f"\n=== Dataset: {dataset} | Generator: DPGAN (synthcity) ===")

            # 1) Real data
            X_real = load_train_df(dataset)

            # 2) Instantiate DPGAN.
            #    dp_epsilon and dp_delta control the privacy level.
            plugin = Plugins().get(
                "dpgan",
                n_iter=N_ITERS,
                epsilon=DP_EPSILON,  # privacy budget
                delta=DP_DELTA,  # optional, can also be set to None
                dp_max_grad_norm=2.0,  # optional
                dp_secure_mode=False  # optional
            )

            # 3) Fit on real data.
            plugin.fit(X_real)

            # 4) generate
            n_target = len(X_real) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
            if MAX_SYNTH_CAP:
                n_target = min(n_target, MAX_SYNTH_CAP)

            synth_df = plugin.generate(n_target)
            print(f"  [info] generated {len(synth_df)} synthetic samples")

            # 5) Align columns to the real dataset.
            for c in X_real.columns:
                if c not in synth_df.columns:
                    synth_df[c] = np.nan
            synth_df = synth_df[X_real.columns]

            # ---------- LABELING RF ----------
            try:
                rf = load_rf_model(dataset)
                X_rf = align_to_model_feature_names(synth_df, rf)
                y_rf = rf.predict(X_rf)

                Xtr, Xte, ytr, yte = train_test_split(
                    X_rf, y_rf, test_size=TEST_SIZE,
                    random_state=RANDOM_STATE, stratify=y_rf
                )
                save_split(dataset, "dpgan", "bbrf", Xtr, Xte, ytr, yte)
                print("  [OK] RF labels generated and saved")
            except Exception as e:
                print(f"  [WARN] RF labeling skipped: {e}")

            # ---------- LABELING NN ----------
            try:
                nn_model = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS))
                expected_in = get_expected_in_features(nn_model)
                X_nn = build_X_for_nn(X_real, synth_df, expected_in)
                y_nn = predict_nn_in_batches(nn_model, X_nn, batch_size=BATCH_SIZE)

                Xtr, Xte, ytr, yte = train_test_split(
                    X_nn, y_nn, test_size=TEST_SIZE,
                    random_state=RANDOM_STATE, stratify=y_nn
                )
                save_split(dataset, "dpgan", "bbnn", Xtr, Xte, ytr, yte)
                print("  [OK] NN labels generated and saved")
            except Exception as e:
                print(f"  [WARN] NN labeling skipped: {e}")

            # ---------- save raw synthetic data (optional) ----------
            out_raw = DIR_SYNTH / dataset / f"synth_dpgan_{dataset}.csv"
            out_raw.parent.mkdir(parents=True, exist_ok=True)
            synth_df.to_csv(out_raw, index=False)
            print(f"  [OK] raw synthetic data saved to {out_raw}")

        except Exception as e:
            print(f"[ERR] {dataset}: {e}")


if __name__ == "__main__":
    main()
