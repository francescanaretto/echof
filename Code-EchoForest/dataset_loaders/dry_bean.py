#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump as joblib_dump


def find_label_column(df: pd.DataFrame) -> str:

    for c in ["Class", "class", "label", "Label", "target", "Target", "y", "Y"]:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find the label column. Expected one of: Class/class/label/target/y. "
        f"Columns found: {list(df.columns)[:30]} ..."
    )


def main():

    in_path = './DryBeanDataset/Dry_Bean_Dataset.xlsx'
    out_dir = '../Data-original/drybean'

    # ---- load
    df = pd.read_excel(in_path, header=0)

    # drop Unnamed
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    # ---- label
    y_col = find_label_column(df)
    y_raw = df[y_col].copy()
    X = df.drop(columns=[y_col]).copy()


    X = X.apply(pd.to_numeric, errors="coerce")


    X = X.dropna(axis=1, how="all")

    mask_ok = ~X.isna().any(axis=1)
    if not mask_ok.all():
        dropped = int((~mask_ok).sum())
        print(f"[warn] Dropping {dropped} rows with NaN values in the features.")
        X = X.loc[mask_ok].reset_index(drop=True)
        y_raw = y_raw.loc[mask_ok].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.to_numpy())

    print(f"[info] label column: {y_col}")
    print(f"[info] classes ({len(le.classes_)}): {list(le.classes_)}")
    vals, cnt = np.unique(y, return_counts=True)
    print(f"[info] class counts: {dict(zip(vals.tolist(), cnt.tolist()))}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    Xtr.to_csv(out_dir +"train_set_drybean.csv")
    Xte.to_csv(out_dir + "test_set_drybean.csv")

    pd.DataFrame({"label": ytr}).to_csv(out_dir + "train_labels_drybean.csv", index=False)
    pd.DataFrame({"label": yte}).to_csv(out_dir + "test_labels_drybean.csv", index=False)

    joblib_dump(le, out_dir + "label_encoder_drybean.joblib")

    print(" -", (out_dir+"train_set_drybean.csv"))
    print(" -", (out_dir+ "test_set_drybean.csv"))
    print(" -", (out_dir+ "train_labels_drybean.csv"))
    print(" -", (out_dir +"test_labels_drybean.csv"))
    print(" -", (out_dir + "label_encoder_drybean.joblib"))


if __name__ == "__main__":
    main()
