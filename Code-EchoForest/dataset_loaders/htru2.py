#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

# CONFIG
DATA_DIR = Path("./htru2")      # directory containing the original file
OUT_DIR  = Path("../Data-original/htru2")

CSV_FILE  = DATA_DIR / "HTRU_2.csv"     # CSV filename
ARFF_FILE = DATA_DIR / "HTRU_2.arff"    # optional

TEST_SIZE = 0.20
RANDOM_STATE = 42

# LOAD DATA
def load_from_csv(path: Path):
    print("[info] Loading CSV without header")
    df = pd.read_csv(path, header=None)
    return df

def load_from_arff(path: Path):
    print("[info] Loading ARFF")
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    return df

# Choose the available source file automatically.
if CSV_FILE.exists():
    df = load_from_csv(CSV_FILE)
elif ARFF_FILE.exists():
    df = load_from_arff(ARFF_FILE)
else:
    raise FileNotFoundError("Could not find either CSV or ARFF.")

# CLEAN + SPLIT
# Last column = label
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

# For ARFF inputs, labels may be stored as bytes.
if y.dtype == object:
    y = y.apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# Force numeric labels 0/1.
y = pd.factorize(y)[0]

print("[info] Total shape:", X.shape)
print("[info] Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

# Stratified split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Save outputs.
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save features with consistent numeric column headers.
feature_cols = [f"f{i}" for i in range(X.shape[1])]

X_train_df = pd.DataFrame(X_train.values, columns=feature_cols)
X_test_df  = pd.DataFrame(X_test.values, columns=feature_cols)

X_train_df.to_csv(OUT_DIR / "train_set_htru2.csv")
X_test_df.to_csv(OUT_DIR / "test_set_htru2.csv")

pd.DataFrame(y_train, columns=["label"]).to_csv(OUT_DIR / "train_labels_htru2.csv")
pd.DataFrame(y_test,  columns=["label"]).to_csv(OUT_DIR / "test_labels_htru2.csv")

print("\n[OK] HTRU2 dataset ready.")
print("Train:", X_train.shape)
print("Test :", X_test.shape)
