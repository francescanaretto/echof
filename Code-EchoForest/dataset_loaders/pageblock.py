#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIG
DATASET_NAME = "wine"

# Path where the UCI files were downloaded.
IN_DIR = Path("../Data-original/wine")  # adjust if needed, e.g. Path("../raw-data/wine")
WINE_DATA_PATH = IN_DIR / "wine.data"

# Output directory in the project format.
OUT_DIR = Path("../Data-original/wine") / DATASET_NAME

TEST_SIZE = 0.20
RANDOM_STATE = 42

# LOAD + SPLIT + SAVE
def main():
    if not WINE_DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {WINE_DATA_PATH.resolve()}")

    # wine.data: first column = class, then 13 numeric features
    df = pd.read_csv(WINE_DATA_PATH, header=None)

    if df.shape[1] < 2:
        raise ValueError(f"{WINE_DATA_PATH} seems to have too few columns: shape={df.shape}")

    y = df.iloc[:, 0].astype(int).to_numpy()
    X = df.iloc[:, 1:].astype(np.float32)

    # Standard feature names.
    X.columns = [f"f{i}" for i in range(X.shape[1])]

    # Stratified split.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Xtr.to_csv(OUT_DIR / f"train_set_{DATASET_NAME}.csv", index=True)
    Xte.to_csv(OUT_DIR / f"test_set_{DATASET_NAME}.csv", index=True)

    pd.DataFrame({"label": ytr}).to_csv(OUT_DIR / f"train_labels_{DATASET_NAME}.csv", index=True)
    pd.DataFrame({"label": yte}).to_csv(OUT_DIR / f"test_labels_{DATASET_NAME}.csv", index=True)

    # Helpful logging.
    def dist(arr):
        vals, cnt = np.unique(arr, return_counts=True)
        return dict(zip(vals.tolist(), cnt.tolist()))

    print(f"[OK] Saved to: {OUT_DIR.resolve()}")
    print(f"     X_train={Xtr.shape}, X_test={Xte.shape}")
    print(f"     y_train dist={dist(ytr)}")
    print(f"     y_test  dist={dist(yte)}")

if __name__ == "__main__":
    main()
