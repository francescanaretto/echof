#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score

"""
Train the original neural-network black-box with an explicit hyperparameter
grid search.

Model selection is performed with repeated stratified cross-validation over the
search space defined in HYPERGRID, followed by a final training pass with early
stopping.
"""

DATASETS = ["shuttle"]
DIR_DATA   = Path("../Data-original")
DIR_MODELS = Path("../Model-original")

DEVICE = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparameter grid searched by repeated stratified cross-validation.
HYPERGRID = {
    "hidden": [
        (10, 5),
        (5,),
        (16, 8),
    ],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64],
    "epochs": [50],
    "patience": [3],
}
torch.set_num_threads(10)
torch.set_num_interop_threads(10)
N_REPEATS = 2
N_SPLITS  = 5

def score_fn(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.int64), dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden_sizes=(10,5), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    hidden: tuple
    dropout: float

def train_one(model, train_loader, val_loader, cfg: TrainConfig):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val = -np.inf
    last_loss = np.inf
    trigger = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                pred = out.argmax(1).cpu().numpy()
                all_p.append(pred)
                all_y.append(yb.cpu().numpy())
        all_p = np.concatenate(all_p)
        all_y = np.concatenate(all_y)
        val_score = score_fn(all_y, all_p)

        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            trigger = 0
        else:
            trigger += 1
            if trigger >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

def predict(model, loader):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            pred = out.argmax(1).cpu().numpy()
            all_p.append(pred)
            all_y.append(yb.numpy())
    return np.concatenate(all_y), np.concatenate(all_p)

def load_original_split(ds: str):
    ddir = DIR_DATA / ds

    def _read_labels_csv(path: Path) -> np.ndarray:
        df = pd.read_csv(path)
        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
        y = df.iloc[:, -1]
        y = np.asarray(y).reshape(-1)
        if y.dtype.kind not in "iu":
            y = pd.factorize(y)[0]
        return y

    def _read_features_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            df = pd.read_csv(path, header=None)

        keep_cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
        df = df.loc[:, keep_cols]

        for labname in ("class", "label", "target", "y"):
            if labname in df.columns:
                df = df.drop(columns=[labname])

        obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if obj_cols:
            df = pd.get_dummies(df, columns=obj_cols, dummy_na=False)

        try:
            df = df.apply(pd.to_numeric, errors="coerce")
        except:
            pass

        df = df.dropna(axis=1, how="all")

        if df.shape[1] == 0:
            raise ValueError(f"{path} does not contain usable numeric features after cleaning.")

        return df

    Xtr_df = _read_features_csv(ddir / f"train_set_{ds}.csv")
    Xte_df = _read_features_csv(ddir / f"test_set_{ds}.csv")

    ytr_raw = _read_labels_csv(ddir / f"train_labels_{ds}.csv")
    yte_raw = _read_labels_csv(ddir / f"test_labels_{ds}.csv")

    le = LabelEncoder()
    ytr = le.fit_transform(ytr_raw)
    yte = le.transform(yte_raw)

    print(f"[dbg][{ds}] classes={list(le.classes_)}  ytr range={ytr.min()}..{ytr.max()}")

    print(f"[dbg][{ds}] Xtr_df shape={Xtr_df.shape} dtypes={Xtr_df.dtypes.value_counts().to_dict()}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df.values)
    Xte = scaler.transform(Xte_df.values)

    in_features = Xtr.shape[1]
    n_classes = int(np.unique(ytr).size)

    return Xtr, Xte, ytr, yte, in_features, n_classes


def make_loaders(X, y, train_idx, val_idx, batch_size):
    tr_ds = TabDataset(X[train_idx], y[train_idx])
    va_ds = TabDataset(X[val_idx], y[val_idx])
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return tr_ld, va_ld

def grid_search_nn(Xtr, ytr, in_features, n_classes):
    """Run repeated-stratified-CV grid search over HYPERGRID."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    best_cfg = None
    best_cv = -np.inf
    for hidden in HYPERGRID["hidden"]:
        for dropout in HYPERGRID["dropout"]:
            for lr in HYPERGRID["lr"]:
                for wd in HYPERGRID["weight_decay"]:
                    for bs in HYPERGRID["batch_size"]:
                        for epochs in HYPERGRID["epochs"]:
                            for patience in HYPERGRID["patience"]:
                                cfg = TrainConfig(
                                    lr=lr, weight_decay=wd, batch_size=bs, epochs=epochs,
                                    patience=patience, hidden=hidden, dropout=dropout
                                )
                                fold_scores = []
                                for tr_idx, va_idx in cv.split(Xtr, ytr):
                                    model = MLP(in_features, n_classes, hidden_sizes=cfg.hidden, dropout=cfg.dropout).to(DEVICE)
                                    tr_ld, va_ld = make_loaders(Xtr, ytr, tr_idx, va_idx, cfg.batch_size)
                                    val_score = train_one(model, tr_ld, va_ld, cfg)
                                    fold_scores.append(val_score)
                                mean_cv = float(np.mean(fold_scores))
                                if mean_cv > best_cv:
                                    best_cv = mean_cv
                                    best_cfg = cfg
                                    print(f"[NN] new best CV={best_cv:.4f} cfg={best_cfg}")
    return best_cfg, best_cv

def train_full_and_save(ds, Xtr, ytr, Xte, yte, in_features, n_classes, cfg: TrainConfig):
    """Train the final model with the best grid-search configuration."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    tr_idx, va_idx = next(iter(cv.split(Xtr, ytr)))
    tr_ld, va_ld = make_loaders(Xtr, ytr, tr_idx, va_idx, cfg.batch_size)

    model = MLP(in_features, n_classes, hidden_sizes=cfg.hidden, dropout=cfg.dropout).to(DEVICE)
    _ = train_one(model, tr_ld, va_ld, cfg)

    train_loader_full = DataLoader(TabDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=False)
    test_loader_full  = DataLoader(TabDataset(Xte, yte), batch_size=cfg.batch_size, shuffle=False)

    y_tr, p_tr = predict(model, train_loader_full)
    y_te, p_te = predict(model, test_loader_full)

    rep_tr = classification_report(y_tr, p_tr)
    rep_te = classification_report(y_te, p_te)

    outdir = (DIR_MODELS / ds)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / f"nn_{ds}.sav")
    (outdir / f"nn_{ds}_report_train.txt").write_text(rep_tr)
    (outdir / f"nn_{ds}_report_test.txt").write_text(rep_te)
    with open(outdir / f"nn_{ds}_best_params.txt", "w") as f:
        f.write(str(cfg))

    print(f"{ds} train_bal_acc={balanced_accuracy_score(y_tr, p_tr):.4f} "
          f"test_bal_acc={balanced_accuracy_score(y_te, p_te):.4f}")
    return model

def main():
    for ds in DATASETS:
        try:
            print(f"\n NN grid search: {ds}")
            Xtr, Xte, ytr, yte, in_features, n_classes = load_original_split(ds)
            best_cfg, best_cv = grid_search_nn(Xtr, ytr, in_features, n_classes)
            print(f"[{ds}] Best CV={best_cv:.4f} cfg={best_cfg}")

            _ = train_full_and_save(ds, Xtr, ytr, Xte, yte, in_features, n_classes, best_cfg)

        except Exception as e:
            print(f"[{ds}] {e}")

if __name__ == "__main__":
    main()
