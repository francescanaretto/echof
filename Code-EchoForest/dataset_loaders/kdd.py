#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# CONFIG
FILE_PATH = Path("kdd")   # <-- path al file ARFF (senza estensione ok)
TARGET_COL = "class"
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_DIR = Path("../Data-original/kdd")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ARFF loader robusto
def load_arff_simple(path: Path) -> pd.DataFrame:
    columns = []
    rows = []
    data_started = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("%"):
                continue

            low = line.lower()

            if low.startswith("@attribute"):
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    continue
                col = parts[1].strip()
                if col.startswith("'") and col.endswith("'"):
                    col = col[1:-1]
                columns.append(col)
                continue

            if low.startswith("@data"):
                data_started = True
                continue

            if data_started:
                normalized = line.replace("'", '"')
                parsed = next(csv.reader([normalized], delimiter=",", quotechar='"', skipinitialspace=True))
                if len(parsed) == len(columns):
                    rows.append([x.strip() for x in parsed])

    return pd.DataFrame(rows, columns=columns)

# MAIN
def main():
    df = load_arff_simple(FILE_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not found.")

    # Normalize missing values.
    df.replace({"?": pd.NA, "": pd.NA, "nan": pd.NA, "NaN": pd.NA}, inplace=True)

    # Separate X and y.
    X = df.drop(columns=[TARGET_COL])
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Cast to numeric where possible and keep categorical columns as strings.
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="ignore")

    # Use a stratified split when class labels are valid.
    stratify_arg = y
    if y.isna().any() or y.nunique() < 2:
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_arg
    )

    # save
    X_train.to_csv(OUT_DIR / "train_set_kdd.csv", index=False)
    X_test.to_csv(OUT_DIR / "test_set_kdd.csv", index=False)
    y_train.to_csv(OUT_DIR / "train_labels_kdd.csv", index=False)
    y_test.to_csv(OUT_DIR / "test_labels_kdd.csv", index=False)

    print("Train/test files saved successfully in:", OUT_DIR)

if __name__ == "__main__":
    main()
