#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIG
RAW_PATH = Path("./california")
DATASET_NAME = "california"
OUT_DIR = Path("../Data-original") / DATASET_NAME

TEST_SIZE = 0.2
SEED = 42

MAKE_CLASSIFICATION = True
CLASS_MODE = "quantile_k"
K_CLASSES = 5

# HELPERS
_numline = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?(\s+|,|;|\t)")

def looks_numeric_row(s: str) -> bool:

    if not _numline.search(s):
        return False
    toks = re.split(r"[,\s;]+", s.strip())
    nums = 0
    for t in toks:
        if t == "":
            continue
        try:
            float(t)
            nums += 1
        except Exception:
            pass
    return nums >= 2

def detect_start_line(path: Path, max_scan: int = 5000) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            if looks_numeric_row(line):
                return i
    raise RuntimeError("Could not find the start of the numeric rows. Is the file in the expected format?")

def read_numeric_table(path: Path, skiprows: int) -> pd.DataFrame:

    try:
        df = pd.read_csv(path, skiprows=skiprows, header=None, sep=r"\s+", engine="python")
        if df.shape[1] >= 2 and df.notna().sum().sum() > 0:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, skiprows=skiprows, header=None)
        if df.shape[1] >= 2 and df.notna().sum().sum() > 0:
            return df
    except Exception:
        pass

    df = pd.read_csv(path, skiprows=skiprows, header=None, sep=";")
    return df

print(f"[info] reading: {RAW_PATH}")
start = detect_start_line(RAW_PATH)
print(f"[info] detected numeric data starts at line: {start}")

df = read_numeric_table(RAW_PATH, skiprows=start)
df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

print(f"[info] loaded shape: {df.shape}")

if df.shape[1] == 9:
    df.columns = [
        "MedInc","HouseAge","AveRooms","AveBedrms","Population",
        "AveOccup","Latitude","Longitude","MedHouseVal"
    ]

target_col = "MedHouseVal" if "MedHouseVal" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col]).astype(np.float32)
y = df[target_col].astype(np.float32).to_numpy()

print(f"[info] target column: {target_col}")

if MAKE_CLASSIFICATION:
    if CLASS_MODE == "median":
        thr = float(np.median(y))
        y_cls = (y > thr).astype(int)
        print(f"[info] classification: median split thr={thr:.4f} (binary)")
    elif CLASS_MODE == "quantile_k":
        qs = np.quantile(y, np.linspace(0, 1, K_CLASSES + 1))
        qs = np.unique(qs)
        if len(qs) < 3:
            thr = float(np.median(y))
            y_cls = (y > thr).astype(int)
            print(f"[warn] quantili degeneri -> fallback median thr={thr:.4f}")
        else:
            # bins: [q0,q1), [q1,q2), ...
            y_cls = np.digitize(y, bins=qs[1:-1], right=False).astype(int)
            print(f"[info] classification: quantile_k K={K_CLASSES} (labels 0..{K_CLASSES-1})")
    else:
        raise ValueError("CLASS_MODE deve essere 'median' o 'quantile_k'")
else:
    y_cls = y

OUT_DIR.mkdir(parents=True, exist_ok=True)

Xtr, Xte, ytr, yte = train_test_split(
    X, y_cls, test_size=TEST_SIZE, random_state=SEED,
    stratify=y_cls if MAKE_CLASSIFICATION else None
)

Xtr.to_csv(OUT_DIR / f"train_set_{DATASET_NAME}.csv", index=False)
Xte.to_csv(OUT_DIR / f"test_set_{DATASET_NAME}.csv", index=False)
pd.Series(ytr).to_csv(OUT_DIR / f"train_labels_{DATASET_NAME}.csv", index=False)
pd.Series(yte).to_csv(OUT_DIR / f"test_labels_{DATASET_NAME}.csv", index=False)

print(f"[OK] saved in: {OUT_DIR.resolve()}")
print(f"[info] Xtr={Xtr.shape} Xte={Xte.shape}")

if MAKE_CLASSIFICATION:
    vals, cnt = np.unique(ytr, return_counts=True)
    print(f"[info] train class distribution: {dict(zip(vals.tolist(), cnt.tolist()))}")
