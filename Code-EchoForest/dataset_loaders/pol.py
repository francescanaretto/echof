#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize the existing binary pol dataset split under Data-original/pol.
"""

from pathlib import Path

import pandas as pd


DATASET_NAME = "pol"
DATA_DIR = Path("../Data-original") / DATASET_NAME


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    label_col = df.columns[-1]
    return pd.DataFrame({"label": df[label_col].astype(int)})


def main() -> None:
    train_x = DATA_DIR / "train_set_pol.csv"
    test_x = DATA_DIR / "test_set_pol.csv"
    train_y = DATA_DIR / "train_labels_pol.csv"
    test_y = DATA_DIR / "test_labels_pol.csv"

    for path in (train_x, test_x, train_y, test_y):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    xtr = load_features(train_x)
    xte = load_features(test_x)
    ytr = load_labels(train_y)
    yte = load_labels(test_y)

    xtr.to_csv(train_x, index=False)
    xte.to_csv(test_x, index=False)
    ytr.to_csv(train_y, index=False)
    yte.to_csv(test_y, index=False)

    print(f"[OK] Normalized pol dataset files in: {DATA_DIR.resolve()}")
    print(f"[info] train shape: {xtr.shape} | test shape: {xte.shape}")


if __name__ == "__main__":
    main()
