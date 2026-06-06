

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


ARFF_PATH = Path("./breast-cancer-coimbra.arff")
DATASET_NAME = "breast_coimbra"
OUT_BASE = Path("../Data-original") / DATASET_NAME

TEST_SIZE = 0.2
SEED = 42


print("[info] loading ARFF...")

data, meta = arff.loadarff(str(ARFF_PATH))
df = pd.DataFrame(data)

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].apply(
            lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
        )

df = df.replace("?", np.nan)

print(f"[info] dataset shape: {df.shape}")


label_col = df.columns[-1]
print(f"[info] label column: {label_col}")

y_raw = df[label_col]
X_df = df.drop(columns=[label_col])

# One-hot encode categorical columns if present.
cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    X_df = pd.get_dummies(X_df, columns=cat_cols, dummy_na=True)

# Convert everything to numeric.
X_df = X_df.apply(pd.to_numeric, errors="coerce")

# Fill NaN values with the median.
if X_df.isna().any().any():
    X_df = X_df.fillna(X_df.median(numeric_only=True))

# LABEL ENCODING 0..K-1
if y_raw.dtype.kind in "iu":
    y_vals = y_raw.astype(int).to_numpy()
    unique = np.unique(y_vals)
    mapping = {int(v): i for i, v in enumerate(unique)}
    y = np.array([mapping[int(v)] for v in y_vals])
else:
    y, uniques = pd.factorize(y_raw.astype(str), sort=True)
    mapping = {str(uniques[i]): int(i) for i in range(len(uniques))}

print(f"[info] classes found: {mapping}")

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

print(f"[info] train shape: {X_train.shape}")
print(f"[info] test shape:  {X_test.shape}")

# SAVE
OUT_BASE.mkdir(parents=True, exist_ok=True)

X_train.to_csv(OUT_BASE / f"train_set_{DATASET_NAME}.csv", index=False)
X_test.to_csv(OUT_BASE / f"test_set_{DATASET_NAME}.csv", index=False)

pd.Series(y_train).to_csv(
    OUT_BASE / f"train_labels_{DATASET_NAME}.csv", index=False
)
pd.Series(y_test).to_csv(
    OUT_BASE / f"test_labels_{DATASET_NAME}.csv", index=False
)

pd.DataFrame({
    "raw_label": list(mapping.keys()),
    "encoded": list(mapping.values())
}).to_csv(OUT_BASE / f"label_mapping_{DATASET_NAME}.csv", index=False)

print(f"\n[OK] Saved to {OUT_BASE.resolve()}")
