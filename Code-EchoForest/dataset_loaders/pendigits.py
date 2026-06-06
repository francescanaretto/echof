#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

# CONFIG
DATASET_NAME = "pendigits"

# Directory containing the .tra / .tes files.
IN_DIR = Path("./pen+based+recognition")  # adjust if needed
TRAIN_PATH = IN_DIR / "pendigits.tra"
TEST_PATH  = IN_DIR / "pendigits.tes"

# Output directory in the project format.
OUT_DIR = Path("../Data-original") / DATASET_NAME

# Pendigits: 16 features + 1 label (last column).
N_FEATURES = 16

def _load_pendigits(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    # No header, comma-separated file.
    df = pd.read_csv(path, header=None)

    if df.shape[1] != (N_FEATURES + 1):
        raise ValueError(
            f"{path.name}: expected {N_FEATURES+1} columns (16 features + label), "
            f"found {df.shape[1]}. Example shape={df.shape}"
        )

    X = df.iloc[:, :N_FEATURES].copy()
    y = df.iloc[:, N_FEATURES].astype(int).to_numpy()

    # Use feature names consistent with the other datasets.
    X.columns = [f"f{i}" for i in range(N_FEATURES)]

    # Ensure numeric dtype.
    X = X.apply(pd.to_numeric, errors="raise").astype(np.float32)

    return X, y

def _dist(y: np.ndarray) -> dict:
    vals, cnt = np.unique(y, return_counts=True)
    return dict(zip(vals.tolist(), cnt.tolist()))

def main():
    Xtr, ytr = _load_pendigits(TRAIN_PATH)
    Xte, yte = _load_pendigits(TEST_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Xtr.to_csv(OUT_DIR / f"train_set_{DATASET_NAME}.csv", index=True)
    Xte.to_csv(OUT_DIR / f"test_set_{DATASET_NAME}.csv", index=True)

    pd.DataFrame({"label": ytr}).to_csv(OUT_DIR / f"train_labels_{DATASET_NAME}.csv", index=True)
    pd.DataFrame({"label": yte}).to_csv(OUT_DIR / f"test_labels_{DATASET_NAME}.csv", index=True)

    print(f"[OK] Saved to: {OUT_DIR.resolve()}")
    print(f"     X_train={Xtr.shape}, X_test={Xte.shape}")
    print(f"     y_train dist={_dist(ytr)}")
    print(f"     y_test  dist={_dist(yte)}")

if __name__ == "__main__":
    main()
