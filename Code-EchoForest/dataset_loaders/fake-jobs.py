#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from joblib import dump as joblib_dump

# CONFIG
FILE_PATH = Path("fake-jobs")   # <-- path al file ARFF
TARGET_COL = "fraudulent"
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_DIR = Path("../Data-original/fake-jobs")
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
    print("Colonne:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not found.")

    # Remove long free-text columns.
    cols_to_drop = ["title", "description"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Separate X and y.
    X = df.drop(columns=[TARGET_COL])
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # train test split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # =====================
    # ONE HOT ENCODING
    # =====================
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_enc = ohe.fit_transform(X_train)
    X_test_enc = ohe.transform(X_test)

    feature_names = ohe.get_feature_names_out(X.columns)

    X_train_enc_df = pd.DataFrame(X_train_enc, columns=feature_names)
    X_test_enc_df = pd.DataFrame(X_test_enc, columns=feature_names)

    print("Shape after OHE train:", X_train_enc_df.shape)
    print("Shape after OHE test:", X_test_enc_df.shape)

    # =====================
    # SALVATAGGIO
    # =====================

    # versioni raw (senza testo ma non encodate)
    #X_train.to_csv(OUT_DIR / "train_set_fake_jobs.csv", index=False)
    #X_test.to_csv(OUT_DIR / "test_set_fake_jobs.csv", index=False)
    y_train.to_csv(OUT_DIR / "train_labels_fake_jobs.csv", index=False)
    y_test.to_csv(OUT_DIR / "test_labels_fake_jobs.csv", index=False)

    # versioni encodate
    X_train_enc_df.to_csv(OUT_DIR / "train_set_fake_jobs.csv", index=False)
    X_test_enc_df.to_csv(OUT_DIR / "test_set_fake_jobs.csv", index=False)

    # save encoder
    joblib_dump(ohe, OUT_DIR / "ohe_fake_jobs.joblib")

    print("Split and one-hot encoding completed successfully.")


if __name__ == "__main__":
    main()
