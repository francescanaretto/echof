#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from joblib import dump as joblib_dump


FILE_PATH = Path("employee")
TARGET_COL = "LeaveOrNot"
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_DIR = Path("../Data-original/employee")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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



def main():
    df = load_arff_simple(FILE_PATH)

    print("Dataset shape:", df.shape)
    print("Colonne:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not found.")

    df = df.drop_duplicates()
    print("Shape after drop_duplicates:", df.shape)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    for col in ["JoiningYear", "Age", "ExperienceInCurrentDomain", "PaymentTier"]:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])

    X_train_enc = pd.concat(
        [
            X_train[num_cols].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(cat_cols))
        ],
        axis=1
    )

    X_test_enc = pd.concat(
        [
            X_test[num_cols].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(cat_cols))
        ],
        axis=1
    )

    print("Shape after encoding train:", X_train_enc.shape)
    print("Shape after encoding test:", X_test_enc.shape)

    y_train.to_csv(OUT_DIR / "train_labels_employee.csv", index=False)
    y_test.to_csv(OUT_DIR / "test_labels_employee.csv", index=False)

    X_train_enc.to_csv(OUT_DIR / "train_set_employee.csv", index=False)
    X_test_enc.to_csv(OUT_DIR / "test_set_employee.csv", index=False)

    joblib_dump(ohe, OUT_DIR / "ohe_employee.joblib")

    print("Split and one-hot encoding completed for employee.")


if __name__ == "__main__":
    main()
