#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
"""
Normalize the existing binary landsat2 dataset split under Data-original/landsat2.

The script validates the existing train/test files, removes index-like columns,
and rewrites the standard split files with a consistent `label` column.
"""

from pathlib import Path

import pandas as pd

from datasets import load_dataset

ds = load_dataset("mstz/landsat", "landsat")
DATASET_NAME = "landsat2"
DATA_DIR = Path("../Data-original") / DATASET_NAME
print(ds)
ds = ds.to_pandas()
class_counts = ds['class'].value_counts()
print(class_counts)
#print(dataset.shape, dataset['class'])
labels = ds.pop('class')
X_train, X_test, Y_train, Y_test = train_test_split(ds, labels, test_size = 0.2, random_state = 42, stratify=labels)
#print(X_train.shape, X_test.shape, Y_train.shape)

ds.to_csv('../Data-original/landsat2/landsat2.csv')
X_train.to_csv('../Data-original/landsat2/train_set_landsat2.csv')
X_test.to_csv('../Data-original/landsat2/test_set_landsat2.csv')
Y_train.to_csv('../Data-original/landsat2/train_labels_landsat2.csv')
Y_test.to_csv('../Data-original/landsat2/test_labels_landsat2.csv')

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
    train_x = DATA_DIR / "train_set_landsat2.csv"
    test_x = DATA_DIR / "test_set_landsat2.csv"
    train_y = DATA_DIR / "train_labels_landsat2.csv"
    test_y = DATA_DIR / "test_labels_landsat2.csv"

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

    print(f"[OK] Normalized landsat2 dataset files in: {DATA_DIR.resolve()}")
    print(f"[info] train shape: {xtr.shape} | test shape: {xte.shape}")


if __name__ == "__main__":
    main()
