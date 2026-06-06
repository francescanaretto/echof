

import typing
import sys
from pathlib import Path

try:
    from typing_extensions import Self as _Self

    typing.Self = _Self  # type: ignore[attr-defined]
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import json
import math
import pickle
import numpy as np
import pandas as pd
from dpmm.pipelines import MSTPipeline
from sklearn.model_selection import train_test_split
from dpmm.pipelines import AIMPipeline

# Make direct execution from Code/competitors robust after the repository reorganization.
THIS_FILE = Path(__file__).resolve()
CODE_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = CODE_ROOT.parent
for candidate in (CODE_ROOT, CODE_ROOT / "core", CODE_ROOT / "shared"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

# Utility to load the already trained neural networks.
from loaders import load_trained_nn


# generazione DP
DATASETS = ["adult", "activity", "pol", "spotify", "spotify-r", "landsat", "landsat2", "electricity"]
BB_MODELS = ["nn", "rf"]

N_SYNTH   = 80000
MAX_SYNTH_PER_DATASET = 80000
TEST_SIZE = 0.30
RANDOM_STATE = 42
BATCH_SIZE_PRED = 4096
USE_FLOAT32 = True
APPLY_SCALING = True
LOW_CARD_INT_AS_CAT = True
LOW_CARD_THRESHOLD = 30
DIR_DATA   = PROJECT_ROOT / "Data-original"
DIR_SYNTH  = PROJECT_ROOT / "Data-synthetic"
DIR_MODELS = PROJECT_ROOT / "Model-original"
EPSILON = 1.0
PROC_EPSILON = 0.1
DELTA = 1e-5
MAX_MODEL_MB = 80
BINNER = "priv-tree"       # "uniform" / "quantile"

N_ITERS = 1000
DEGREE = 2
NUM_MARGINALS = None
MAX_CELLS = 10_000


LOW_CARD_INT_AS_CAT = True
LOW_CARD_THRESHOLD = 30
NEAR_ZERO_VAR_THR = 1e-12
MAX_CATS_IN_DOMAIN = 1000
N_BINS         = "auto"
BINNER         = "priv-tree"
TEST_SIZE      = 0.30
RANDOM_STATE   = 42
BATCH_SIZE     = 4096

def _to_jsonable(v):

    if isinstance(v, (np.floating,)):
        v = float(v)
    elif isinstance(v, (np.integer,)):
        v = int(v)
    elif isinstance(v, (np.bool_,)):
        v = bool(v)

    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "NaN"
    if v is None:
        return "None"
    return v

def build_pipeline_privbayes(n_bins, degree, n_iters):
    return AIMPipeline(
            epsilon=EPSILON,
            proc_epsilon=PROC_EPSILON,
            binner_type=BINNER,           # "priv-tree"
            gen_kwargs={
                "n_iters": n_iters,
                "degree": degree,
                "num_marginals": NUM_MARGINALS,
                "max_cells": MAX_CELLS,
            },
            delta=DELTA,
            compress=True,
            max_model_size=MAX_MODEL_MB,
            n_bins=n_bins,
        )

def prepare_for_privbayes(df_raw: pd.DataFrame,
                          max_cats: int = 1000) -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()


    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]


    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)

    domain = {"columns": {}, "n_rows": int(len(df))}

    for c in df.columns:
        s = df[c]

        s = s.replace([np.inf, -np.inf], np.nan)


        s_num = pd.to_numeric(s, errors="coerce")
        is_num_candidate = not s_num.isna().all()

        if is_num_candidate:

            unique_vals = s_num.dropna().unique()
            n_unique = len(unique_vals)

            if n_unique <= 1:
                s_cat = s.astype(str)
                s_cat = s_cat.fillna("NaN")
                df[c] = s_cat
                vals = list(pd.unique(s_cat))
                vals = [_to_jsonable(v) for v in vals]
                if len(vals) > max_cats:
                    vals = vals[:max_cats-1] + ["OTHER"]
                domain["columns"][c] = {
                    "type": "categorical",
                    "values": vals
                }
            else:
                med = float(np.nanmedian(s_num.values))
                if not np.isfinite(med):
                    med = 0.0
                s_num = s_num.fillna(med).astype(np.float32)
                df[c] = s_num

                vmin = float(np.nanmin(s_num))
                vmax = float(np.nanmax(s_num))
                if not np.isfinite(vmin): vmin = 0.0
                if not np.isfinite(vmax): vmax = vmin + 1.0

                if abs(vmax - vmin) < 1e-9:
                    vmax = vmin + 1e-6

                domain["columns"][c] = {
                    "type": "continuous",
                    "min": _to_jsonable(vmin),
                    "max": _to_jsonable(vmax)
                }
        else:
            s_cat = s.astype(str)
            s_cat = s_cat.fillna("NaN")
            df[c] = s_cat
            vals = list(pd.unique(s_cat))
            vals = [_to_jsonable(v) for v in vals]
            if len(vals) > max_cats:
                vals = vals[:max_cats-1] + ["OTHER"]
            domain["columns"][c] = {
                "type": "categorical",
                "values": vals
            }

    domain["n_rows"] = int(len(df))
    return df, domain

def load_train_df(dataset: str) -> pd.DataFrame:
    p = DIR_DATA / dataset / f"train_set_{dataset}.csv"
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df

def save_split(dataset: str, kind: str, bb_tag: str, Xtr, Xte, Ytr, Yte):
    outdir = DIR_SYNTH
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{dataset}_{kind}_{bb_tag}"
    pd.DataFrame(Xtr).to_csv(outdir / dataset/ f"train_set_synth_{base}.csv", index=False)
    pd.DataFrame(Xte).to_csv(outdir / dataset/ f"test_set_synth_{base}.csv", index=False)
    pd.Series(Ytr, name="label").to_csv(outdir / dataset/ f"train_labels_synth_{base}.csv", index=False)
    pd.Series(Yte, name="label").to_csv(outdir /dataset/  f"test_labels_synth_{base}.csv", index=False)

def load_rf_model(dataset: str):
    with open(DIR_MODELS / dataset / f"rf_{dataset}.sav", "rb") as f:
        return pickle.load(f)

def align_to_model_feature_names(X: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = list(X.columns)

    X = X.copy()
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        if c.startswith("Unnamed"):
            if np.issubdtype(X.index.dtype, np.number):
                X[c] = X.index.astype(float)
            else:
                X[c] = np.arange(len(X), dtype=float)
        else:
            X[c] = 0
    return X.reindex(columns=expected, fill_value=0)


import torch
import torch.nn as nn

def get_expected_in_features(model: nn.Module) -> int:
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

def build_X_for_nn(Xref_real: pd.DataFrame, X_synth: pd.DataFrame, expected_in: int) -> pd.DataFrame:

    num_cols_real = Xref_real.select_dtypes(include=[np.number]).columns.tolist()
    common = [c for c in num_cols_real if c in X_synth.columns]
    if len(common) < expected_in:
        raise RuntimeError(f"Not enough shared numeric features ({len(common)}) for an NN expecting {expected_in} inputs.")

    X = X_synth[common].copy()

    for c in common:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = pd.to_numeric(Xref_real[c], errors="coerce").median()
        if not np.isfinite(med): med = 0.0
        X[c] = X[c].fillna(med).astype(np.float32)

    return X.iloc[:, :expected_in]

def predict_nn_in_batches(model: nn.Module, X_df: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(X_df), batch_size):
            chunk = X_df.iloc[start:start+batch_size]
            xt = torch.tensor(chunk.to_numpy(dtype=np.float32), dtype=torch.float32)
            logits = model(xt)
            if logits.ndim == 2 and logits.shape[1] > 1:
                pred = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                pred = (torch.sigmoid(logits.flatten()) >= 0.5).cpu().numpy().astype(int)
            out.append(pred)
    return np.concatenate(out, axis=0)


def _to_jsonable(x):
    if isinstance(x, (np.floating,)): x = float(x)
    elif isinstance(x, (np.integer,)): x = int(x)
    elif isinstance(x, (np.bool_,)): x = bool(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return "NaN"
    if x is None: return "None"
    return x

def infer_domain_no_drop(df: pd.DataFrame,
                         low_card_int_as_cat=True,
                         low_card_thr=30,
                         max_cats=1000) -> dict:

    dom = {"columns": {}, "n_rows": int(len(df))}
    for c in df.columns:
        s = df[c]
        s_no_na = s.dropna()
        is_bool = pd.api.types.is_bool_dtype(s)
        is_int  = pd.api.types.is_integer_dtype(s_no_na)
        is_num  = pd.api.types.is_numeric_dtype(s_no_na)
        is_cat  = pd.api.types.is_categorical_dtype(s) or (s_no_na.dtype == object) or is_bool

        if low_card_int_as_cat and is_int and s_no_na.nunique() <= low_card_thr:
            is_cat, is_num = True, False

        if is_cat and not is_num:
            vals = s_no_na.astype(str).unique().tolist()
            vals = [_to_jsonable(v) for v in vals]
            if len(vals) > max_cats:
                vals = vals[:max_cats-1] + ["OTHER"]
            dom["columns"][c] = {"type": "categorical", "values": vals}
        else:
            a = pd.to_numeric(s_no_na, errors="coerce").astype(float)
            if a.empty:
                mn = mx = 0.0
            else:
                mn = float(np.nanmin(a)) if np.isfinite(np.nanmin(a)) else 0.0
                mx = float(np.nanmax(a)) if np.isfinite(np.nanmax(a)) else 0.0
            dom["columns"][c] = {"type": "continuous", "min": mn, "max": mx}
    return dom

def sanitize_no_drop(df: pd.DataFrame, domain: dict) -> tuple[pd.DataFrame, dict]:
    """Does not remove columns: converts bool->int, inf->NaN, and lightly imputes NaN values to avoid crashes."""
    X = df.copy()
    # cast booleani
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
    # inf → NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # all-NaN -> riempi 0 / "NaN"
    for c in X.columns:
        if X[c].isna().all():
            X[c] = 0.0 if pd.api.types.is_numeric_dtype(X[c]) else "NaN"
    domain["n_rows"] = int(len(X))
    return X, domain



def main():
    DIR_SYNTH.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        try:
            print(f"\n=== Dataset: {dataset} | Generator: PrivBayes (DP) ===")
            if dataset == 'adult' or dataset == 'activity' or dataset == 'pol':
                X_real = load_train_df(dataset)

                X_sanit, domain = prepare_for_privbayes(X_real)
            else:
                X_real = load_train_df(dataset)

                domain = infer_domain_no_drop(X_real)
                X_sanit, domain = sanitize_no_drop(X_real, domain)


            pipeline = AIMPipeline(
                epsilon=EPSILON,
                proc_epsilon=PROC_EPSILON,
                binner_type=BINNER,  # "priv-tree"
                gen_kwargs={
                    "n_iters": N_ITERS,
                    "degree": DEGREE,
                    "num_marginals": NUM_MARGINALS,
                    "max_cells": MAX_CELLS,
                },
                delta=DELTA,
                compress=True,
                max_model_size=MAX_MODEL_MB,
                n_bins=N_BINS,
            )
            try:
                pipeline.fit(X_sanit, domain)
                n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                print('prima di generate')
                synth_df = pipeline.generate(n_target)
                print('ho generato', synth_df.shape, X_real.shape)
            except Exception as e:
                print('eccezione della pipeline fit')
                msg = str(e)
                if "probabilities contain NaN" in msg:
                    print('il mex quello di sempre')
                    print("  [warn] PrivBayes returned NaN on the main configuration; trying a more conservative fallback")

                    cols = list(X_sanit.columns)
                    if len(cols) > 20:
                        keep = cols[:20]
                        X_fb = X_sanit[keep].copy()
                        dom_fb = {"columns": {c: domain["columns"][c] for c in keep},
                                  "n_rows": domain["n_rows"]}
                    else:
                        X_fb = X_sanit
                        dom_fb = domain

                    pipeline = build_pipeline_privbayes(
                        n_bins=16,
                        degree=1,
                        n_iters=1000,
                    )

                    pipeline.fit(X_fb, dom_fb)
                    X_sanit = X_fb
                    domain = dom_fb
                    n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                    print('prima di generate')
                    synth_df = pipeline.generate(n_target)
                    print('ho generato', synth_df.shape, X_real.shape)
                else:
                    print('errore diverso!')
                    cols = list(X_sanit.columns)
                    if len(cols) > 20:
                        keep = cols[:20]
                        X_fb = X_sanit[keep].copy()
                        dom_fb = {"columns": {c: domain["columns"][c] for c in keep},
                                  "n_rows": domain["n_rows"]}
                    else:
                        X_fb = X_sanit
                        dom_fb = domain
                    pipeline = build_pipeline_privbayes(
                        n_bins=10,
                        degree=1,
                        n_iters=1000,
                    )
                    pipeline.fit(X_fb, dom_fb)
                    X_sanit = X_fb
                    domain = dom_fb
                    n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                    print('prima di generate')
                    synth_df = pipeline.generate(n_target)
                    print('ho generato', synth_df.shape, X_real.shape)


            for c in X_real.columns:
                if c not in synth_df.columns:
                    synth_df[c] = np.nan
            synth_df = synth_df[X_real.columns]

            try:
                rf = load_rf_model(dataset)
                X_rf = align_to_model_feature_names(synth_df, rf)   # gestisce Unnamed:* e ordine
                y_rf = rf.predict(X_rf)
                Xtr, Xte, ytr, yte = train_test_split(X_rf, y_rf, test_size=TEST_SIZE,
                                                      random_state=RANDOM_STATE, stratify=y_rf)
                save_split(dataset, "privbayes", "bbrf", Xtr, Xte, ytr, yte)
                print("  [OK] RF labels generated and saved")
            except Exception as e:
                print(f"  [WARN] RF labeling skipped: {e}")

            try:
                nn_model = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS))
                expected_in = get_expected_in_features(nn_model)
                X_nn = build_X_for_nn(X_real, synth_df, expected_in)  # use shared numeric columns
                y_nn = predict_nn_in_batches(nn_model, X_nn, batch_size=BATCH_SIZE)
                Xtr, Xte, ytr, yte = train_test_split(X_nn, y_nn, test_size=TEST_SIZE,
                                                      random_state=RANDOM_STATE, stratify=y_nn)
                save_split(dataset, "aim", "bbnn", Xtr, Xte, ytr, yte)
                print("  [OK] NN labels generated and saved")
            except Exception as e:
                print(f"  [WARN] NN labeling skipped: {e}")

            out_raw = DIR_SYNTH/ f"synth_aim_{dataset}.csv"
            synth_df.to_csv(out_raw, index=False)

        except Exception as e:
            print(f"[ERR] {dataset}: {e}")

if __name__ == "__main__":
    main()
