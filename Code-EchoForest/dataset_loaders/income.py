#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the Folktables income dataset under the publication-facing name
`income`.

This loader uses the already prepared splits stored in `Data-original/adult/`,
which correspond to the subset of Folktables states used in the experiments.
The output is written to `Data-original/income/` with the standard file names:

- train_set_income.csv
- test_set_income.csv
- train_labels_income.csv
- test_labels_income.csv

If the prepared split files are not available, the script falls back to
`Adult_2014.csv`, assuming the last column is the label.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_DIR = Path("../Data-original/adult")
TARGET_DIR = Path("../Data-original/income")

SEED = 42
TEST_SIZE = 0.20


def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if len(df.columns) > 0 and str(df.columns[0]) == "":
        df = df.iloc[:, 1:]
    return df


def clean_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if "labels" in df.columns:
        y = df["labels"]
    elif "label" in df.columns:
        y = df["label"]
    else:
        y = df.iloc[:, -1]
    return pd.DataFrame({"label": y.astype(int)})


def copy_existing_split() -> bool:
    train_x = SOURCE_DIR / "train_set_adult.csv"
    test_x = SOURCE_DIR / "test_set_adult.csv"
    train_y = SOURCE_DIR / "train_labels_adult.csv"
    test_y = SOURCE_DIR / "test_labels_adult.csv"

    if not all(p.exists() for p in (train_x, test_x, train_y, test_y)):
        return False

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    xtr = clean_feature_frame(pd.read_csv(train_x))
    xte = clean_feature_frame(pd.read_csv(test_x))
    ytr = clean_label_frame(pd.read_csv(train_y))
    yte = clean_label_frame(pd.read_csv(test_y))

    xtr.to_csv(TARGET_DIR / "train_set_income.csv", index=False)
    xte.to_csv(TARGET_DIR / "test_set_income.csv", index=False)
    ytr.to_csv(TARGET_DIR / "train_labels_income.csv", index=False)
    yte.to_csv(TARGET_DIR / "test_labels_income.csv", index=False)

    return True


def build_from_fallback_csv() -> None:
    source_csv = SOURCE_DIR / "Adult_2014.csv"
    if not source_csv.exists():
        raise FileNotFoundError(
            "Could not find prepared adult splits or fallback Adult_2014.csv."
        )

    df = pd.read_csv(source_csv)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if "labels" not in df.columns:
        raise ValueError("Fallback Adult_2014.csv does not contain a 'labels' column.")

    x = df.drop(columns=["labels"])
    y = df["labels"].astype(int)

    xtr, xte, ytr, yte = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    xtr.to_csv(TARGET_DIR / "train_set_income.csv", index=False)
    xte.to_csv(TARGET_DIR / "test_set_income.csv", index=False)
    pd.DataFrame({"label": ytr}).to_csv(TARGET_DIR / "train_labels_income.csv", index=False)
    pd.DataFrame({"label": yte}).to_csv(TARGET_DIR / "test_labels_income.csv", index=False)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if copy_existing_split():
        print(f"[OK] income dataset created from prepared adult splits in: {TARGET_DIR.resolve()}")
        return

    build_from_fallback_csv()
    print(f"[OK] income dataset created from Adult_2014.csv in: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
