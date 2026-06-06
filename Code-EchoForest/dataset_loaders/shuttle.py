#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# CONFIG
DATASET_NAME = "shuttle"
OUT_DIR = Path("../Data-original/shuttle") / DATASET_NAME
TEST_SIZE = 0.2
RANDOM_STATE = 42

# LOAD DATA
print("[info] Download Shuttle dataset from OpenML...")
X, y = fetch_openml("shuttle", version=1, as_frame=True, return_X_y=True)

print(f"[info] Raw shape: X={X.shape}, y={y.shape}")

# Ensure the features are numeric.
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Convert labels to integers 0..K-1.
if y.dtype.kind not in "iu":
    y, uniques = pd.factorize(y)
else:
    y = y.astype(int)

print(f"[info] Number of classes: {len(np.unique(y))}")

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"[info] Train shape: {X_train.shape}")
print(f"[info] Test  shape: {X_test.shape}")

# SAVE
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_train.to_csv(OUT_DIR / "train_set_shuttle.csv")
X_test.to_csv(OUT_DIR / "test_set_shuttle.csv")

pd.Series(y_train, name="label").to_csv(
    OUT_DIR / "train_labels_shuttle.csv"
)

pd.Series(y_test, name="label").to_csv(
    OUT_DIR / "test_labels_shuttle.csv"
)

print(f"[OK] Dataset saved to {OUT_DIR.resolve()}")
