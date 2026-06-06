#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CONFIG (NO ARGS)
DATASET_NAME = "internet-firewall"
RAW_PATH = Path("./internet-firewall")  # adjust here if needed
OUT_DIR = Path("../Data-original") / DATASET_NAME

TEST_SIZE = 0.2
SEED = 42
MAX_ROWS = None  # None = use all rows

# HELPERS
_num_re = re.compile(r"^[\s,;]*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?([\s,;]+[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)*[\s,;]*$")

def is_data_line(line: str) -> bool:
    """Return True if the line looks like a numeric data row."""
    s = line.strip()
    if not s:
        return False
    # Exclude lines that look like metadata or headers.
    bad_prefix = ("@", "#", "%", "//")
    if s.startswith(bad_prefix):
        return False
    # If the line contains letters, it is often a header.
    # Here we check whether the feature portion looks numeric.
    # Fully numeric rows are always accepted.
    if _num_re.match(s):
        return True
    return False

def detect_delim(sample_line: str) -> str:
    """Return the separator regex for pandas.read_csv: comma/semicolon or whitespace."""
    s = sample_line.strip()
    if "," in s:
        return r"\s*,\s*"
    if ";" in s:
        return r"\s*;\s*"
    # default whitespace (spazi o tab)
    return r"\s+"

def find_first_data_row(path: Path, max_scan: int = 5000) -> tuple[int, str]:
    """Find the index of the first data row and return a sample line."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            if is_data_line(line):
                return i, line
    # Fallback: if no fully numeric row is found, use the first non-empty line.
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip():
                return i, line
    raise RuntimeError("File is empty or unreadable.")

# LOAD
print(f"[info] reading: {RAW_PATH}")
skiprows, sample = find_first_data_row(RAW_PATH)
sep = detect_delim(sample)

print(f"[info] detected data start at line: {skiprows}")
print(f"[info] detected separator: {'comma/semicolon' if sep != r'\\s+' else 'whitespace'}")

# Read the data block without a header.
df = pd.read_csv(
    RAW_PATH,
    header=None,
    skiprows=skiprows,
    sep=sep,
    engine="python",
    comment=None,
)

# Drop fully empty columns, which can happen with messy separators.
df = df.dropna(axis=1, how="all")
print(f"[info] loaded raw table shape: {df.shape}")

if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)

# Last column = label.
y_raw = df.iloc[:, -1]
X_df = df.iloc[:, :-1].copy()

# Convert features to numeric and map NaN to 0.
X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_df = X_df.astype(np.float32)

# Label-encode the target, even if it is already numeric.
le = LabelEncoder()
y = le.fit_transform(np.asarray(y_raw).reshape(-1))

OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"original_label": le.classes_, "encoded_label": np.arange(len(le.classes_))}) \
  .to_csv(OUT_DIR / f"label_mapping_{DATASET_NAME}.csv", index=False)

print(f"[info] n_classes={len(le.classes_)}")

# SPLIT + SAVE
Xtr, Xte, ytr, yte = train_test_split(
    X_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

Xtr.to_csv(OUT_DIR / f"train_set_{DATASET_NAME}.csv", index=False)
Xte.to_csv(OUT_DIR / f"test_set_{DATASET_NAME}.csv", index=False)
pd.Series(ytr).to_csv(OUT_DIR / f"train_labels_{DATASET_NAME}.csv", index=False)
pd.Series(yte).to_csv(OUT_DIR / f"test_labels_{DATASET_NAME}.csv", index=False)

vals, cnt = np.unique(ytr, return_counts=True)
print(f"[OK] saved in: {OUT_DIR.resolve()}")
print(f"[info] Xtr={Xtr.shape} Xte={Xte.shape}")
print(f"[info] train class distribution: {dict(zip(vals.tolist(), cnt.tolist()))}")
