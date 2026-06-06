#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typing

try:
    from typing_extensions import Self as _Self

    typing.Self = _Self  # type: ignore[attr-defined]
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# dpmm: PrivBayes
from dpmm.pipelines import PrivBayesPipeline

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

# ============ CONFIG ============
DATASETS   = [ "pol"]
DIR_DATA   = PROJECT_ROOT / "Data-original"
DIR_MODELS = PROJECT_ROOT / "Model-original"
DIR_SYNTH  = PROJECT_ROOT / "Data-synthetic"

# generazione DP
EPSILON        = 1.0
PROC_EPSILON   = 0.1
DELTA          = 1e-5
N_BINS         = "auto"        # deve rimanere "priv-tree"
BINNER         = "priv-tree"
MAX_MODEL_MB   = 80
N_ITERS        = 5000
DEGREE         = 2
N_SYNTH        = 80000          # or "match" to use len(train_set)
MAX_SYNTH_CAP  = 80000

TEST_SIZE      = 0.30
RANDOM_STATE   = 42
BATCH_SIZE     = 4096

ROBUST_PRIVBAYES_DATASETS = {
    "adult",
    "activity",
    "pol",
    "wave-multi",
    "wave_multi",
}


def _to_jsonable(v):
    """Safe value for JSON and dpmm: no None and no NaN/inf."""
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
    return PrivBayesPipeline(
        epsilon=EPSILON,
        proc_epsilon=PROC_EPSILON,
        binner_type=BINNER,   # "priv-tree" come vuoi tu
        gen_kwargs={
            "n_iters": n_iters,
            "degree": degree,
        },
        delta=DELTA,
        compress=True,
        max_model_size=MAX_MODEL_MB,
        n_bins=n_bins,
    )


def prepare_wave_multi_privbayes(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    More conservative preprocessing for wave-multi style datasets.

    The goal is to avoid non-monotonic bin edges inside the DPMM binning stage by
    handing PrivBayes an already discretized categorical representation.
    """
    df = df_raw.copy()
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df = df.replace([np.inf, -np.inf], np.nan)

    x_out = pd.DataFrame(index=df.index)
    domain = {"columns": {}, "n_rows": int(len(df))}

    for c in df.columns:
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.isna().all():
            cats = pd.Series(["0"] * len(df), index=df.index, dtype="object")
        else:
            med = float(np.nanmedian(s_num.to_numpy(dtype=float)))
            if not np.isfinite(med):
                med = 0.0
            s_num = s_num.fillna(med)

            nunique = int(pd.Series(s_num).nunique(dropna=True))
            if nunique <= 1:
                cats = pd.Series(np.zeros(len(s_num), dtype=int), index=df.index).astype(str)
            else:
                q = max(2, min(10, nunique))
                try:
                    cats = pd.qcut(s_num, q=q, labels=False, duplicates="drop")
                except Exception:
                    cats = pd.cut(s_num, bins=q, labels=False, include_lowest=True, duplicates="drop")
                cats = pd.Series(cats, index=df.index).fillna(0).astype(int).astype(str)

        x_out[c] = cats
        uniq = sorted(x_out[c].unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else x)
        domain["columns"][c] = {"type": "categorical", "values": uniq}

    return x_out, domain

def prepare_for_privbayes(df_raw: pd.DataFrame,
                          max_cats: int = 1000) -> tuple[pd.DataFrame, dict]:
    """
    Ritorna:
      - df_clean: DataFrame senza NaN/inf/None (numeriche float32, categoriche string)
      - domain:   dizionario dominio dpmm senza None/NaN nei valori
    Regole:
      - removes columns 'Unnamed:*'
      - bool -> int
      - numeriche:
          * se n_unique <= 1 -> trattata come categorica (label string)
          * otherwise: impute NaN/inf with the median and use the continuous range [min,max]
      - non-numeriche -> categoriche (string)
    """
    df = df_raw.copy()

    # 0) rimuovi eventuali "Unnamed:*"
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # 1) booleani -> int
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)

    domain = {"columns": {}, "n_rows": int(len(df))}

    for c in df.columns:
        s = df[c]
        # sostituisci inf con NaN per poter imputare
        s = s.replace([np.inf, -np.inf], np.nan)

        # prova a vederla come numerica
        s_num = pd.to_numeric(s, errors="coerce")
        is_num_candidate = not s_num.isna().all()

        if is_num_candidate:
            # true numeric column: decide whether to treat it as continuous or categorical
            unique_vals = s_num.dropna().unique()
            n_unique = len(unique_vals)

            if n_unique <= 1:
                # constant -> categorical
                # castiamo tutto a stringa e imputiamo NaN con "NaN"
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
                # true continuous column: impute NaN with the median
                med = float(np.nanmedian(s_num.values))
                if not np.isfinite(med):
                    med = 0.0
                s_num = s_num.fillna(med).astype(np.float32)
                df[c] = s_num

                vmin = float(np.nanmin(s_num))
                vmax = float(np.nanmax(s_num))
                if not np.isfinite(vmin): vmin = 0.0
                if not np.isfinite(vmax): vmax = vmin + 1.0

                # se range degenerato, allarghiamo un minimo
                if abs(vmax - vmin) < 1e-9:
                    vmax = vmin + 1e-6

                domain["columns"][c] = {
                    "type": "continuous",
                    "min": _to_jsonable(vmin),
                    "max": _to_jsonable(vmax)
                }
        else:
            # non-numeric: treat it as purely categorical
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
    # Read the full feature matrix and only drop explicit saved-index columns.
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

# ---- sklearn pipeline alignment (RF) ----
def align_to_model_feature_names(X: pd.DataFrame, model) -> pd.DataFrame:
    """Riallinea X ai nomi/ordine visti in fit (gestisce Unnamed:0 mancante)."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = list(X.columns)

    X = X.copy()
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        if c.startswith("Unnamed"):
            # recreate from the index as a proxy
            if np.issubdtype(X.index.dtype, np.number):
                X[c] = X.index.astype(float)
            else:
                X[c] = np.arange(len(X), dtype=float)
        else:
            X[c] = 0
    return X.reindex(columns=expected, fill_value=0)

# ---- utilities for NN (PyTorch) ----
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
    """
    Costruisce un DataFrame numerico (float32) per la NN:
    - use ONLY numeric columns from real data that also exist in the synthetic data
    - simple imputation (real-data median)
    - taglia/ordina al numero atteso di input
    """
    num_cols_real = Xref_real.select_dtypes(include=[np.number]).columns.tolist()
    common = [c for c in num_cols_real if c in X_synth.columns]
    if len(common) < expected_in:
        raise RuntimeError(f"Not enough shared numeric features ({len(common)}) for an NN expecting {expected_in} inputs.")

    X = X_synth[common].copy()
    # conversione & imputazione
    for c in common:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = pd.to_numeric(Xref_real[c], errors="coerce").median()
        if not np.isfinite(med): med = 0.0
        X[c] = X[c].fillna(med).astype(np.float32)

    # deterministically keep the first expected_in columns
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

# ---- dominio minimale per dpmm (senza rimozioni) ----
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
    """
    Create a dpmm domain without removing columns.
    - low-cardinality integers -> categorical
    - string/object/bool -> categorical
    - numeriche -> continuous con (min,max)
    """
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


# ============ MAIN LOOP ============

def main():
    DIR_SYNTH.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        try:
            print(f"\n=== Dataset: {dataset} | Generator: PrivBayes (DP) ===")
            if dataset in ROBUST_PRIVBAYES_DATASETS:
                X_real = load_train_df(dataset)

                # 2) preparazione robusta per PrivBayes
                if dataset in {"wave-multi", "wave_multi"}:
                    X_sanit, domain = prepare_wave_multi_privbayes(X_real)
                else:
                    X_sanit, domain = prepare_for_privbayes(X_real)
            else:
                # 1) Real data
                X_real = load_train_df(dataset)

                # 2) dominio dpmm (senza drop) e sanitizzazione minima
                domain = infer_domain_no_drop(X_real)
                X_sanit, domain = sanitize_no_drop(X_real, domain)


            # 3) pipeline PrivBayes
            pipeline = PrivBayesPipeline(
                epsilon=EPSILON,
                proc_epsilon=PROC_EPSILON,
                binner_type=BINNER,
                gen_kwargs={"n_iters": N_ITERS, "degree": DEGREE},
                delta=DELTA,
                compress=True,
                max_model_size=MAX_MODEL_MB,
                n_bins=N_BINS
            )
            try:
                pipeline.fit(X_sanit, domain)
                n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                n_target = min(n_target, MAX_SYNTH_CAP) if MAX_SYNTH_CAP else n_target
                print('prima di generate')
                synth_df = pipeline.generate(n_target)
                print('ho generato', synth_df.shape, X_real.shape)
            except Exception as e:
                print('eccezione della pipeline fit')
                msg = str(e)
                if "probabilities contain NaN" in msg:
                    print('il mex quello di sempre')
                    print("  [warn] PrivBayes returned NaN on the main configuration; trying a more conservative fallback")

                    # (optional) reduce dimensionality slightly (drop 5 less informative columns)
                    cols = list(X_sanit.columns)
                    if len(cols) > 20:
                        # ad es.: tieni solo le prime 20 (deterministico)
                        keep = cols[:20]
                        X_fb = X_sanit[keep].copy()
                        dom_fb = {"columns": {c: domain["columns"][c] for c in keep},
                                  "n_rows": domain["n_rows"]}
                    else:
                        X_fb = X_sanit
                        dom_fb = domain

                    # pipeline di fallback: bins fissi, grado 1, meno iterazioni
                    pipeline = build_pipeline_privbayes(
                        n_bins=16,  # fixed, no "auto"
                        degree=1,  # meno interazioni
                        n_iters=1000,  # meno rumorosa / meno rischi
                    )

                    pipeline.fit(X_fb, dom_fb)
                    X_sanit = X_fb
                    domain = dom_fb
                    n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                    n_target = min(n_target, MAX_SYNTH_CAP) if MAX_SYNTH_CAP else n_target
                    print('prima di generate')
                    synth_df = pipeline.generate(n_target)
                    print('ho generato', synth_df.shape, X_real.shape)
                elif "bins must increase monotonically" in msg and dataset in {"wave-multi", "wave_multi"}:
                    print("  [warn] non-monotonic bins on wave-multi – retrying with pre-discretized categories")

                    X_fb, dom_fb = prepare_wave_multi_privbayes(X_real)
                    pipeline = build_pipeline_privbayes(
                        n_bins=10,
                        degree=1,
                        n_iters=1000,
                    )

                    pipeline.fit(X_fb, dom_fb)
                    X_sanit = X_fb
                    domain = dom_fb
                    n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                    n_target = min(n_target, MAX_SYNTH_CAP) if MAX_SYNTH_CAP else n_target
                    print('prima di generate')
                    synth_df = pipeline.generate(n_target)
                    print('ho generato', synth_df.shape, X_real.shape)
                else:
                    # errore diverso: lo propago
                    print('errore diverso!')
                    cols = list(X_sanit.columns)
                    if len(cols) > 20:
                        # ad es.: tieni solo le prime 20 (deterministico)
                        keep = cols[:20]
                        X_fb = X_sanit[keep].copy()
                        dom_fb = {"columns": {c: domain["columns"][c] for c in keep},
                                  "n_rows": domain["n_rows"]}
                    else:
                        X_fb = X_sanit
                        dom_fb = domain
                    pipeline = build_pipeline_privbayes(
                        n_bins=10,  # fixed, no "auto"
                        degree=1,  # meno interazioni
                        n_iters=1000,  # meno rumorosa / meno rischi
                    )
                    pipeline.fit(X_fb, dom_fb)
                    X_sanit = X_fb
                    domain = dom_fb
                    n_target = len(X_sanit) if str(N_SYNTH).lower() == "match" else int(N_SYNTH)
                    n_target = min(n_target, MAX_SYNTH_CAP) if MAX_SYNTH_CAP else n_target
                    print('prima di generate')
                    synth_df = pipeline.generate(n_target)
                    print('ho generato', synth_df.shape, X_real.shape)

            # 4) fit + generate
            #pipeline.fit(X_sanit, domain)


            # 5) column alignment (same set and order as the real data)
            for c in X_real.columns:
                if c not in synth_df.columns:
                    synth_df[c] = np.nan
            synth_df = synth_df[X_real.columns]

            # ---------- LABEL CON RF ----------
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

            # ---------- LABEL CON NN ----------
            try:
                nn_model = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS))
                expected_in = get_expected_in_features(nn_model)
                X_nn = build_X_for_nn(X_real, synth_df, expected_in)  # use shared numeric columns
                y_nn = predict_nn_in_batches(nn_model, X_nn, batch_size=BATCH_SIZE)
                Xtr, Xte, ytr, yte = train_test_split(X_nn, y_nn, test_size=TEST_SIZE,
                                                      random_state=RANDOM_STATE, stratify=y_nn)
                save_split(dataset, "privbayes", "bbnn", Xtr, Xte, ytr, yte)
                print("  [OK] NN labels generated and saved")
            except Exception as e:
                print(f"  [WARN] NN labeling skipped: {e}")

            # 6) save raw synthetic data (optional)
            out_raw = DIR_SYNTH/ f"synth_privbayes_{dataset}.csv"
            synth_df.to_csv(out_raw, index=False)

        except Exception as e:
            print(f"[ERR] {dataset}: {e}")

if __name__ == "__main__":
    main()
