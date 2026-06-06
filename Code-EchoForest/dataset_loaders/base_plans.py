

import csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump as joblib_dump


FILE_PATH = Path("base-plans")
DATASET_NAME = "base-plans"

TARGET_COL = "target"
CAT_COLS = ["sourcing_channel", "residence_area_type"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_DIR = Path(f"../Data-original/{DATASET_NAME}")
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

    df.replace({"?": pd.NA, "": pd.NA, "nan": pd.NA, "NaN": pd.NA}, inplace=True)

    for c in CAT_COLS:
        if c not in df.columns:
            raise KeyError(f"Categorical column '{c}' not found.")
        df[c] = df[c].astype("string")
        df[c] = df[c].fillna("__MISSING__").str.strip()

    # X / y
    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype("string").fillna("__MISSING__"))

    for c in X.columns:
        if c not in CAT_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train.to_csv(OUT_DIR / f"train_set_{DATASET_NAME}.csv", index=False)
    X_test.to_csv(OUT_DIR / f"test_set_{DATASET_NAME}.csv", index=False)
    pd.Series(y_train, name=TARGET_COL).to_csv(OUT_DIR / f"train_labels_{DATASET_NAME}.csv", index=False)
    pd.Series(y_test, name=TARGET_COL).to_csv(OUT_DIR / f"test_labels_{DATASET_NAME}.csv", index=False)

    joblib_dump(le, OUT_DIR / f"label_encoder_{DATASET_NAME}.joblib")

    print("Train/test files saved successfully in:", OUT_DIR)
    print("✔ Classi target (encoder):", list(le.classes_))

if __name__ == "__main__":
    main()
