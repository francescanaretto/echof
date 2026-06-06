

import pandas as pd
from pathlib import Path
from joblib import dump as joblib_dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff

# CONFIG
FILE_PATH = Path("churn.arff")
DATASET_NAME = "churn"

TARGET_COL = "class"
CAT_COLS = [
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_DIR = Path(f"../Data-original/{DATASET_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpers
def load_arff_to_df(path: Path) -> pd.DataFrame:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
            except Exception:
                pass
    return df

def fit_transform_label_encoders(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str]
):
    encoders = {}

    for c in cat_cols:
        if c not in X_train.columns:
            raise KeyError(f"Categorical column '{c}' not found in X.")

        tr = X_train[c].astype("string").fillna("__MISSING__").str.strip()
        te = X_test[c].astype("string").fillna("__MISSING__").str.strip()

        le = LabelEncoder()
        le.fit(tr)

        mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
        X_train[c] = tr.map(mapping).astype(int)
        X_test[c] = te.map(mapping).fillna(-1).astype(int)

        encoders[c] = le

    return X_train, X_test, encoders

def main():
    df = load_arff_to_df(FILE_PATH)

    print("Dataset shape:", df.shape)
    print("Colonne:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not found. Imposta TARGET_COL correttamente.")

    df.replace({"?": pd.NA, "": pd.NA, "nan": pd.NA, "NaN": pd.NA}, inplace=True)

    # X / y
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    if y.dtype == object or str(y.dtype).startswith("string"):
        y_le = LabelEncoder()
        y = y.astype("string").fillna("__MISSING__").str.strip()
        y = y_le.fit_transform(y)
        joblib_dump(y_le, OUT_DIR / f"label_encoder_target_{DATASET_NAME}.joblib")
        print("[info] Encoded target saved (label_encoder_target).")
    else:
        y = pd.to_numeric(y, errors="coerce")

    stratify_arg = y if pd.Series(y).nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_arg
    )

    if CAT_COLS:
        X_train, X_test, encoders = fit_transform_label_encoders(X_train, X_test, CAT_COLS)
        joblib_dump(encoders, OUT_DIR / f"label_encoders_{DATASET_NAME}.joblib")
        print(f"[info] Salvati label encoders per: {CAT_COLS}")
    else:
        encoders = {}
        print("[warn] CAT_COLS is empty: no encoded feature was produced.")

    for c in X_train.columns:
        if c not in CAT_COLS:
            X_train[c] = pd.to_numeric(X_train[c], errors="ignore")
            X_test[c] = pd.to_numeric(X_test[c], errors="ignore")

    X_train.to_csv(OUT_DIR / f"train_set_{DATASET_NAME}.csv", index=False)
    X_test.to_csv(OUT_DIR / f"test_set_{DATASET_NAME}.csv", index=False)
    pd.Series(y_train, name=TARGET_COL).to_csv(OUT_DIR / f"train_labels_{DATASET_NAME}.csv", index=False)
    pd.Series(y_test, name=TARGET_COL).to_csv(OUT_DIR / f"test_labels_{DATASET_NAME}.csv", index=False)

    print("Train/test files saved successfully in:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
