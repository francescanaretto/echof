#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from pathlib import Path
import sys

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# NEW (attacker model / LiRA utils)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Make the Code root importable regardless of the current working directory.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from loaders import load_trained_nn

# Configuration
DATASETS   = ["adult", "activity", "spotify", "spotify-r", "pol"]

KIND       = "entropy"
GUIDING_BB = "nn"          # "rf" | "nn"
PERCENTILE = 25

# Which surrogate to attack:
# - "standard": rf_{dataset}_{kind}_{guiding_bb}_{percentile}_wise.sav
# - "dpquery":  rf_dpquery_{dataset}_{kind}_{guiding_bb}_{percentile}_{epsilon}_{mech}_{noise}_wise.sav
SURROGATE_MODE = "standard"
DPQUERY_EPSILON = 0.5
DPQUERY_MECH = "laplace"  # "laplace" | "gaussian"
# Set to True/False. Use False for the "no flag"/legacy runs.
DPQUERY_NOISE_ON_LABELING = False

PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DIR_SYNTH        = (PROJECT_ROOT / "Data-synthetic" / "wise").resolve()
DIR_MODELS_BB    = (PROJECT_ROOT / "Model-original").resolve()
DIR_MODELS_STUD  = (PROJECT_ROOT / "Model-synthetic-wise").resolve()
DIR_ORIG_SPLITS  = (PROJECT_ROOT / "Data-original").resolve()

# LiRA shadows
DIR_SHADOW_MODELS = (PROJECT_ROOT / "Model-shadow").resolve()
N_SHADOW_MAX      = 32

# NEW: toggles
DO_FEATURE_ATTACK = True
DO_LIRA_ATTACK    = True
DO_GRAD_NORM      = True    # NN only

RNG_SEED          = 42
DEVICE            = torch.device("cpu")

# AUTO-CREATE SHADOWS (LiRA)
AUTO_CREATE_SHADOWS = True
N_SHADOW_CREATE     = 10      # create up to this many per dataset (if missing)
SHADOW_EPOCHS       = 10
SHADOW_LR           = 1e-3
SHADOW_BATCH        = 512
SHADOW_SUBSAMPLE    = 0.5     # fraction of target-train used as IN per shadow
MIN_SHADOWS_FOR_LIRA = 5

# NN predict
@torch.no_grad()
def predict_proba_nn_in_batches(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size]
        xt = torch.tensor(chunk, dtype=torch.float32, device=DEVICE)
        logits = model(xt)
        if logits.ndim == 2 and logits.shape[1] > 1:
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            p1 = torch.sigmoid(logits.flatten()).cpu().numpy()
            proba = np.stack([1 - p1, p1], axis=1)
        out.append(proba)
    return np.concatenate(out, axis=0)

# Loss per-sample (robusta)
def per_sample_ce_with_classs(proba, y, classs, eps=1e-12):
    proba = np.clip(np.asarray(proba), eps, 1.0)
    y = np.asarray(y).reshape(-1).astype(int)
    classs = np.asarray(classs).reshape(-1).astype(int)

    class_to_col = {int(c): i for i, c in enumerate(classs)}
    missing = [int(v) for v in np.unique(y) if int(v) not in class_to_col]
    if missing:
        raise ValueError(f"Label non presenti in classs: {missing} | classs={classs.tolist()}")

    cols = np.array([class_to_col[int(yi)] for yi in y], dtype=int)
    return -np.log(proba[np.arange(len(y)), cols])

def per_sample_ce_with_classs_rf(proba: np.ndarray, y, classs) -> np.ndarray:
    proba = np.asarray(proba)
    y = np.asarray(y).reshape(-1).astype(int)
    classs = np.asarray(classs).reshape(-1).astype(int)
    class_to_col = {int(c): i for i, c in enumerate(classs)}
    missing = [int(v) for v in np.unique(y) if int(v) not in class_to_col]
    if missing:
        raise ValueError(f"Label non presenti in rf.classs_: {missing} | rf.classs_={classs.tolist()}")
    y_idx = np.array([class_to_col[int(yy)] for yy in y], dtype=int)
    p = np.clip(proba[np.arange(len(y)), y_idx], 1e-12, 1.0)
    return -np.log(p)

# Model helpers
def load_rf_model(dataset: str):
    with open(DIR_ORIG_SPLITS / dataset / f"rf_{dataset}.sav", "rb") as f:
        return pickle.load(f)

def load_trained(dataset: str, base_dir: str):
    if GUIDING_BB == "rf":
        bb = load_rf_model(dataset)
    else:
        bb = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS_BB))
        bb.to(DEVICE)
        bb.eval()
    return bb

def try_load_nn_scaler(dataset: str):
    return None

# RF synth loading
def load_rf_synth(dataset: str, kind: str, guiding_bb: str, percentile: int):
    if SURROGATE_MODE == "dpquery":
        candidates = [
            DIR_MODELS_STUD
            / dataset
            / f"rf_dpquery_{dataset}_{kind}_{guiding_bb}_{percentile}_{DPQUERY_EPSILON}_{DPQUERY_MECH}_{DPQUERY_NOISE_ON_LABELING}_wise.sav",
            DIR_MODELS_STUD
            / dataset
            / f"rf_dpquery_{dataset}_{kind}_{guiding_bb}_{percentile}_{DPQUERY_EPSILON}_{DPQUERY_MECH}_{DPQUERY_NOISE_ON_LABELING}.sav",
        ]
        # Legacy naming without the boolean flag.
        if DPQUERY_NOISE_ON_LABELING is False:
            candidates.extend(
                [
                    DIR_MODELS_STUD
                    / dataset
                    / f"rf_dpquery_{dataset}_{kind}_{guiding_bb}_{percentile}_{DPQUERY_EPSILON}_{DPQUERY_MECH}_wise.sav",
                    DIR_MODELS_STUD
                    / dataset
                    / f"rf_dpquery_{dataset}_{kind}_{guiding_bb}_{percentile}_{DPQUERY_EPSILON}_{DPQUERY_MECH}.sav",
                ]
            )
        model_path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        model_path = DIR_MODELS_STUD / dataset / f"rf_{dataset}_{kind}_{guiding_bb}_{percentile}_wise.sav"
    if not model_path.exists():
        raise FileNotFoundError(f"RF synthetic not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Data loading (labels robust)
def _read_labels_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    for cand in ["label", "labels", "y", "target", "class"]:
        if cand in df.columns:
            s = df[cand]
            break
    else:
        s = df.iloc[:, -1]

    s = s.astype(str).str.strip()
    s = s[s.str.lower() != "labels"]  # drop header-ish row if present

    y_num = pd.to_numeric(s, errors="coerce")
    if y_num.notna().all():
        return y_num.astype(int).to_numpy().reshape(-1)

    uniq = sorted(s.unique().tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    return s.map(mapping).astype(int).to_numpy().reshape(-1)

def load_original_split_csv(dataset: str):
    dd = DIR_ORIG_SPLITS / dataset
    Xtr = pd.read_csv(dd / f"train_set_{dataset}.csv")
    Xte = pd.read_csv(dd / f"test_set_{dataset}.csv")

    Xtr = Xtr.loc[:, ~Xtr.columns.str.startswith("Unnamed")]
    Xte = Xte.loc[:, ~Xte.columns.str.startswith("Unnamed")]

    ytr = _read_labels_csv(dd / f"train_labels_{dataset}.csv")
    yte = _read_labels_csv(dd / f"test_labels_{dataset}.csv")

    if len(ytr) != len(Xtr):
        raise ValueError(f"len(ytr)={len(ytr)} != len(Xtr)={len(Xtr)} for {dataset}")
    if len(yte) != len(Xte):
        raise ValueError(f"len(yte)={len(yte)} != len(Xte)={len(Xte)} for {dataset}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr.to_numpy(dtype=np.float32))
    Xte = scaler.transform(Xte.to_numpy(dtype=np.float32))

    return Xtr, Xte, ytr, yte

# Score-based attacks
def mia_loss_attack_scores(losse.g. np.ndarray) -> np.ndarray:
    return -np.asarray(losses).reshape(-1)

def mia_confidence_attack_scores(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    return np.max(proba, axis=1)

def mia_entropy_attack_scores(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    proba = np.clip(np.asarray(proba), eps, 1.0)
    H = -np.sum(proba * np.log(proba), axis=1)
    return -H

def mia_margin_attack_scores(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    sorted_p = -np.sort(-proba, axis=1)
    p1 = sorted_p[:, 0]
    p2 = sorted_p[:, 1] if sorted_p.shape[1] > 1 else np.zeros_like(p1)
    return p1 - p2

def metrics_from_scores(score.g. np.ndarray, membership: np.ndarray):
    scores = np.asarray(scores).reshape(-1)
    membership = np.asarray(membership).reshape(-1).astype(int)
    aucv = roc_auc_score(membership, scores)
    fpr, tpr, _ = roc_curve(membership, scores)
    target_fpr = 0.01
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    tpr_at_1 = tpr[max(idx, 0)]
    return {"auc": float(aucv), "tpr@fpr=1%": float(tpr_at_1)}

# ATTACK 1: Feature-vector attacker (LogReg)
def build_feature_matrix(proba: np.ndarray, y: np.ndarray, classs: np.ndarray):
    y = np.asarray(y).reshape(-1).astype(int)
    proba = np.asarray(proba)

    loss = per_sample_ce_with_classs(proba, y, classs)
    top1 = np.max(proba, axis=1)
    ent = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)

    sorted_p = -np.sort(-proba, axis=1)
    margin = sorted_p[:, 0] - (sorted_p[:, 1] if proba.shape[1] > 1 else 0.0)

    pred = np.argmax(proba, axis=1)

    class_to_col = {int(c): i for i, c in enumerate(np.asarray(classs).reshape(-1).astype(int))}
    y_idx = np.array([class_to_col[int(yy)] for yy in y], dtype=int)
    p_true = proba[np.arange(len(y)), y_idx]
    correct = (pred == y).astype(float)

    Xfeat = np.column_stack([loss, top1, ent, margin, correct, p_true]).astype(np.float32)
    return Xfeat

def run_feature_vector_attack(proba: np.ndarray, y: np.ndarray, membership: np.ndarray, tag: str):
    classs = np.sort(np.unique(np.asarray(y).reshape(-1).astype(int)))
    Xfeat = build_feature_matrix(proba, y, classs)
    m = np.asarray(membership).reshape(-1).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    oof = np.zeros(len(m), dtype=float)

    for tr_idx, te_idx in skf.split(Xfeat, m):
        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
        clf.fit(Xfeat[tr_idx], m[tr_idx])
        oof[te_idx] = clf.predict_proba(Xfeat[te_idx])[:, 1]

    res = metrics_from_scores(oof, m)
    print(f"\n--- MIA (FEATURE-ATTACK / {tag}) ---")
    print(f"{tag}: AUC={res['auc']:.4f} | TPR@1%FPR={res['tpr@fpr=1%']:.4f}")
    return res

# LiRA: auto-create shadows + true IN/OUT with masks
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _infer_n_classs(y: np.ndarray) -> int:
    y = np.asarray(y).reshape(-1).astype(int)
    k = int(np.unique(y).size)
    return max(2, k)

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, n_classs: int):
        super().__init__()
        h1 = max(32, min(256, 4 * in_dim))
        h2 = max(32, min(256, 2 * in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_classs),
        )

    def forward(self, x):
        return self.net(x)

def _train_shadow_mlp(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch: int) -> nn.Module:
    model = TabularMLP(X.shape[1], _infer_n_classs(y)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(np.asarray(y).reshape(-1).astype(int), dtype=torch.long, device=DEVICE)

    n = X_t.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb = X_t[idx]
            yb = y_t[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model

def _shadow_paths(dataset: str, sid: int):
    d = DIR_SHADOW_MODELS / dataset
    return (d / f"shadow_{sid:03d}.pt", d / f"shadow_{sid:03d}_mask.npy")

def _find_existing_shadows(dataset: str, max_n: int) -> list[int]:
    ids = []
    for sid in range(max_n):
        p_model, p_mask = _shadow_paths(dataset, sid)
        if p_model.exists() and p_mask.exists():
            ids.append(sid)
    return ids

def _load_shadow_with_mask(dataset: str, sid: int):
    p_model, p_mask = _shadow_paths(dataset, sid)
    if not (p_model.exists() and p_mask.exists()):
        return None, None
    model = torch.load(p_model, map_location=DEVICE)
    if not isinstance(model, nn.Module):
        raise RuntimeError(f"{p_model} is not a torch nn.Module")
    mask = np.load(p_mask)
    model.to(DEVICE).eval()
    return model, mask

def create_shadows_if_missing(dataset: str, X_all: np.ndarray, y_all: np.ndarray, train_size: int):
    """
    Creates shadows and masks over X_all.
    IN set for each shadow is a random subset of target-train indices [0..train_size-1].
    Mask is boolean over X_all indicating IN for that shadow.
    """
    _ensure_dir(DIR_SHADOW_MODELS / dataset)

    existing = _find_existing_shadows(dataset, N_SHADOW_MAX)
    if len(existing) >= N_SHADOW_CREATE:
        return

    # choose next shadow id after the last existing
    next_id = (max(existing) + 1) if existing else 0
    to_make = min(N_SHADOW_CREATE - len(existing), max(0, N_SHADOW_MAX - next_id))

    if to_make <= 0:
        return

    print(f"[LiRA] Creating {to_make} shadow models for {dataset} (existing={len(existing)})...")

    in_pool = np.arange(train_size)
    for j in range(to_make):
        sid = next_id + j
        p_model, p_mask = _shadow_paths(dataset, sid)
        if p_model.exists() and p_mask.exists():
            continue

        rng = np.random.default_rng(RNG_SEED + sid)
        k_in = max(2, int(len(in_pool) * SHADOW_SUBSAMPLE))
        idx_in = rng.choice(in_pool, size=k_in, replace=False)

        mask = np.zeros(len(y_all), dtype=bool)
        mask[idx_in] = True

        X_in = X_all[idx_in].astype(np.float32, copy=False)
        y_in = y_all[idx_in].astype(int, copy=False)

        model = _train_shadow_mlp(X_in, y_in, epochs=SHADOW_EPOCHS, lr=SHADOW_LR, batch=SHADOW_BATCH)

        torch.save(model, p_model)
        np.save(p_mask, mask)
        print(f"[shadow] saved {p_model.name} + {p_mask.name} | IN={mask.sum()}")

def lira_scores_from_shadows_true(dataset: str, X: np.ndarray, y: np.ndarray):
    """
    True LiRA (with per-shadow IN/OUT masks).
    Returns per-example Gaussian params (mu_in, sd_in, mu_out, sd_out).
    """
    y = np.asarray(y).reshape(-1).astype(int)
    classs = np.sort(np.unique(y))

    shadow_ids = _find_existing_shadows(dataset, N_SHADOW_MAX)
    loaded = 0

    losses_in  = [[] for _ in range(len(y))]
    losses_out = [[] for _ in range(len(y))]

    for sid in shadow_ids:
        sm, mask_in = _load_shadow_with_mask(dataset, sid)
        if sm is None:
            continue
        proba_s = predict_proba_nn_in_batches(sm, X)
        loss_s  = per_sample_ce_with_classs(proba_s, y, classs)

        # fill IN/OUT lists
        mask_in = np.asarray(mask_in).astype(bool)
        if mask_in.shape[0] != len(y):
            raise ValueError(f"Shadow mask length mismatch: {mask_in.shape[0]} vs n={len(y)}")

        in_idx = np.where(mask_in)[0]
        out_idx = np.where(~mask_in)[0]

        for i in in_idx:
            losses_in[i].append(float(loss_s[i]))
        for i in out_idx:
            losses_out[i].append(float(loss_s[i]))

        loaded += 1

    if loaded < MIN_SHADOWS_FOR_LIRA:
        print(f"\n[LiRA] Shadows with masks found for {dataset}: {loaded}. Need >= {MIN_SHADOWS_FOR_LIRA}; skipping LiRA.")
        return None

    mu_in  = np.zeros(len(y), dtype=np.float32)
    sd_in  = np.zeros(len(y), dtype=np.float32)
    mu_out = np.zeros(len(y), dtype=np.float32)
    sd_out = np.zeros(len(y), dtype=np.float32)

    for i in range(len(y)):
        lin  = np.array(losses_in[i], dtype=np.float32)
        lout = np.array(losses_out[i], dtype=np.float32)

        # fallback if a sample never appears IN (or OUT)
        if lin.size == 0 and lout.size == 0:
            # should not happen if masks cover all samples; but safe fallback
            mu_in[i] = 0.0; sd_in[i] = 1.0
            mu_out[i] = 0.0; sd_out[i] = 1.0
            continue
        if lin.size == 0:
            lin = lout
        if lout.size == 0:
            lout = lin

        mu_in[i]  = float(lin.mean())
        sd_in[i]  = float(lin.std() + 1e-6)
        mu_out[i] = float(lout.mean())
        sd_out[i] = float(lout.std() + 1e-6)

    print(f"\n[LiRA] Loaded shadows={loaded} (true IN/OUT with masks).")
    return (mu_in, sd_in, mu_out, sd_out)

def lira_score_from_params(loss_target, mu_in, sd_in, mu_out, sd_out):
    loss_target = np.asarray(loss_target).reshape(-1)

    def logpdf(x, mu, sd):
        return -0.5*np.log(2*np.pi*sd*sd) - 0.5*((x-mu)/sd)**2

    return logpdf(loss_target, mu_in, sd_in) - logpdf(loss_target, mu_out, sd_out)

# ATTACK 3: Gradient-norm attack (NN only, white-box)
def gradient_norm_scores_nn(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
    model.eval()
    y = np.asarray(y).reshape(-1).astype(int)
    n = len(y)

    # find last Linear
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No nn.Linear found for gradient-norm attack.")

    scores = np.zeros(n, dtype=np.float32)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=DEVICE, requires_grad=False)
        yb = torch.tensor(y[start:end], dtype=torch.long, device=DEVICE, requires_grad=False)

        logits = model(xb)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.shape[1] == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)

        losses = loss_fn(logits, yb)

        for i in range(end - start):
            model.zero_grad(set_to_none=True)
            losses[i].backward(retain_graph=True)

            gn = 0.0
            if last_linear.weight.grad is not None:
                gn += float(torch.sum(last_linear.weight.grad**2).detach().cpu())
            if last_linear.bias is not None and last_linear.bias.grad is not None:
                gn += float(torch.sum(last_linear.bias.grad**2).detach().cpu())
            scores[start + i] = np.sqrt(gn)

    return scores

# MAIN
def main():
    for dataset in DATASETS:
        print(f"\n=== DATASET: {dataset} ===")

        # 1) Load original split
        Xtr_raw, Xte_raw, ytr, yte = load_original_split_csv(dataset)

        # membership vector: train=1, test=0
        X_all = np.vstack([Xtr_raw, Xte_raw])
        y_all = np.concatenate([ytr, yte]).astype(int).reshape(-1)
        m_all = np.concatenate([np.ones(len(ytr), dtype=bool), np.zeros(len(yte), dtype=bool)])

        classs_y = np.sort(np.unique(y_all))

        # 2) Load NN original (+ scaler se serve)
        nn_model = load_trained(dataset=dataset, base_dir=str(DIR_MODELS_BB))

        scaler = try_load_nn_scaler(dataset)
        X_nn = X_all if scaler is None else scaler.transform(X_all)

        # NN proba + score-based
        proba_nn = predict_proba_nn_in_batches(nn_model, X_nn)
        loss_nn = per_sample_ce_with_classs(proba_nn, y_all, classs_y)

        scores_nn_loss = mia_loss_attack_scores(loss_nn)
        scores_nn_conf = mia_confidence_attack_scores(proba_nn)
        scores_nn_ent  = mia_entropy_attack_scores(proba_nn)
        scores_nn_mar  = mia_margin_attack_scores(proba_nn)

        res_nn_loss = metrics_from_scores(scores_nn_loss, m_all)
        res_nn_conf = metrics_from_scores(scores_nn_conf, m_all)
        res_nn_ent  = metrics_from_scores(scores_nn_ent,  m_all)
        res_nn_mar  = metrics_from_scores(scores_nn_mar,  m_all)

        # 3) Load RF synthetic
        rf = load_rf_synth(dataset, KIND, GUIDING_BB, PERCENTILE)
        proba_rf = rf.predict_proba(X_all)
        loss_rf  = per_sample_ce_with_classs_rf(proba_rf, y_all, rf.classs_)

        scores_rf_loss = mia_loss_attack_scores(loss_rf)
        scores_rf_conf = mia_confidence_attack_scores(proba_rf)
        scores_rf_ent  = mia_entropy_attack_scores(proba_rf)
        scores_rf_mar  = mia_margin_attack_scores(proba_rf)

        res_rf_loss = metrics_from_scores(scores_rf_loss, m_all)
        res_rf_conf = metrics_from_scores(scores_rf_conf, m_all)
        res_rf_ent  = metrics_from_scores(scores_rf_ent,  m_all)
        res_rf_mar  = metrics_from_scores(scores_rf_mar,  m_all)

        # Include mode in the filename to avoid overwriting runs (and to ease downstream parsing).
        if SURROGATE_MODE == "dpquery":
            report_path = (
                f"privacy_report_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}"
                f"_dpquery_{DPQUERY_EPSILON}_{DPQUERY_MECH}_{DPQUERY_NOISE_ON_LABELING}.txt"
            )
        else:
            report_path = f"privacy_report_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}.txt"
        with open(report_path, "a") as f:
            # ----------------------------
            # Feature-vector attacker
            # ----------------------------
            if DO_FEATURE_ATTACK:
                _ = run_feature_vector_attack(proba_nn, y_all, m_all, tag="NN")
                _ = run_feature_vector_attack(proba_rf, y_all, m_all, tag="RFsyn")

            # ----------------------------
            # LiRA (auto-create shadows if missing)
            # ----------------------------
            if DO_LIRA_ATTACK:
                if AUTO_CREATE_SHADOWS:
                    existing = _find_existing_shadows(dataset, N_SHADOW_MAX)
                    if len(existing) < MIN_SHADOWS_FOR_LIRA:
                        create_shadows_if_missing(dataset, X_nn, y_all, train_size=len(ytr))

                params = lira_scores_from_shadows_true(dataset, X_nn, y_all)
                if params is not None:
                    mu_in, sd_in, mu_out, sd_out = params
                    scores_lira_nn = lira_score_from_params(loss_nn, mu_in, sd_in, mu_out, sd_out)
                    res_lira_nn = metrics_from_scores(scores_lira_nn, m_all)
                    f.write("\n--- MIA (LiRA / NN, true masks) ---\n")
                    f.write(f"NN LiRA: AUC={res_lira_nn['auc']:.4f} | TPR@1%FPR={res_lira_nn['tpr@fpr=1%']:.4f}\n")
                    print("\n--- MIA (LiRA / NN, true masks) ---")
                    print(f"NN LiRA: AUC={res_lira_nn['auc']:.4f} | TPR@1%FPR={res_lira_nn['tpr@fpr=1%']:.4f}")

            # ----------------------------
            # Gradient-norm attack (NN only)
            # ----------------------------
            if DO_GRAD_NORM:
                try:
                    scores_gn = gradient_norm_scores_nn(nn_model, X_nn, y_all, batch_size=256)
                    res_gn_pos = metrics_from_scores(scores_gn, m_all)
                    res_gn_neg = metrics_from_scores(-scores_gn, m_all)
                    print("\n--- MIA (GRAD-NORM / NN) ---")
                    f.write("\n--- MIA (GRAD-NORM / NN) ---\n")
                    f.write(f"grad-norm:     AUC={res_gn_pos['auc']:.4f} | TPR@1%FPR={res_gn_pos['tpr@fpr=1%']:.4f}\n")
                    f.write(f"-grad-norm:    AUC={res_gn_neg['auc']:.4f} | TPR@1%FPR={res_gn_neg['tpr@fpr=1%']:.4f}\n")
                    print(f"grad-norm:     AUC={res_gn_pos['auc']:.4f} | TPR@1%FPR={res_gn_pos['tpr@fpr=1%']:.4f}")
                    print(f"-grad-norm:    AUC={res_gn_neg['auc']:.4f} | TPR@1%FPR={res_gn_neg['tpr@fpr=1%']:.4f}")
                except Exception as e:
                    print(f"[warn] Gradient-norm attack skipped for {dataset}: {e}")

            # ----------------------------
            # Existing report (kept)
            # ----------------------------
            f.write("\n--- Membership Inference (LOSS attack) ---\n")
            f.write(f"NN original:   AUC={res_nn_loss['auc']:.4f} | TPR@1%FPR={res_nn_loss['tpr@fpr=1%']:.4f}\n")
            f.write(f"RF synthetic:   AUC={res_rf_loss['auc']:.4f} | TPR@1%FPR={res_rf_loss['tpr@fpr=1%']:.4f}\n")

            f.write("\n--- Membership Inference (CONFIDENCE attack) ---\n")
            f.write(f"NN original:   AUC={res_nn_conf['auc']:.4f} | TPR@1%FPR={res_nn_conf['tpr@fpr=1%']:.4f}\n")
            f.write(f"RF synthetic:   AUC={res_rf_conf['auc']:.4f} | TPR@1%FPR={res_rf_conf['tpr@fpr=1%']:.4f}\n")

            f.write("\n--- Membership Inference (ENTROPY attack) ---\n")
            f.write(f"NN original:   AUC={res_nn_ent['auc']:.4f} | TPR@1%FPR={res_nn_ent['tpr@fpr=1%']:.4f}\n")
            f.write(f"RF synthetic:   AUC={res_rf_ent['auc']:.4f} | TPR@1%FPR={res_rf_ent['tpr@fpr=1%']:.4f}\n")

            f.write("\n--- Membership Inference (MARGIN attack) ---\n")
            f.write(f"NN original:   AUC={res_nn_mar['auc']:.4f} | TPR@1%FPR={res_nn_mar['tpr@fpr=1%']:.4f}\n")
            f.write(f"RF synthetic:   AUC={res_rf_mar['auc']:.4f} | TPR@1%FPR={res_rf_mar['tpr@fpr=1%']:.4f}\n")

            f.write("\n--- Gap (NN - RF) ---\n")
            f.write(f"[LOSS]   ΔAUC={res_nn_loss['auc'] - res_rf_loss['auc']:.4f} | ΔTPR@1%={res_nn_loss['tpr@fpr=1%'] - res_rf_loss['tpr@fpr=1%']:.4f}\n")
            f.write(f"[CONF]   ΔAUC={res_nn_conf['auc'] - res_rf_conf['auc']:.4f} | ΔTPR@1%={res_nn_conf['tpr@fpr=1%'] - res_rf_conf['tpr@fpr=1%']:.4f}\n")
            f.write(f"[ENT]    ΔAUC={res_nn_ent['auc']  - res_rf_ent['auc'] :.4f} | ΔTPR@1%={res_nn_ent['tpr@fpr=1%']  - res_rf_ent['tpr@fpr=1%'] :.4f}\n")
            f.write(f"[MARG]   ΔAUC={res_nn_mar['auc']  - res_rf_mar['auc'] :.4f} | ΔTPR@1%={res_nn_mar['tpr@fpr=1%']  - res_rf_mar['tpr@fpr=1%'] :.4f}\n")

        # Console summary
        print("\n--- Membership Inference (LOSS attack) ---")
        print(f"NN original:   AUC={res_nn_loss['auc']:.4f} | TPR@1%FPR={res_nn_loss['tpr@fpr=1%']:.4f}")
        print(f"RF synthetic:   AUC={res_rf_loss['auc']:.4f} | TPR@1%FPR={res_rf_loss['tpr@fpr=1%']:.4f}")

        print("\n--- Membership Inference (CONF / ENT / MAR) ---")
        print(f"NN CONF: AUC={res_nn_conf['auc']:.4f} | TPR@1%FPR={res_nn_conf['tpr@fpr=1%']:.4f}")
        print(f"NN ENT:  AUC={res_nn_ent['auc']:.4f} | TPR@1%FPR={res_nn_ent['tpr@fpr=1%']:.4f}")
        print(f"NN MAR:  AUC={res_nn_mar['auc']:.4f} | TPR@1%FPR={res_nn_mar['tpr@fpr=1%']:.4f}")
        print(f"RF CONF: AUC={res_rf_conf['auc']:.4f} | TPR@1%FPR={res_rf_conf['tpr@fpr=1%']:.4f}")
        print(f"RF ENT:  AUC={res_rf_ent['auc']:.4f} | TPR@1%FPR={res_rf_ent['tpr@fpr=1%']:.4f}")
        print(f"RF MAR:  AUC={res_rf_mar['auc']:.4f} | TPR@1%FPR={res_rf_mar['tpr@fpr=1%']:.4f}")

        print("\n--- Gap (NN - RFsyn) ---")
        print(f"ΔAUC(LOSS): {res_nn_loss['auc'] - res_rf_loss['auc']:+.6f}")
        print(f"[info] report appended to: {report_path}")

if __name__ == "__main__":
    main()
