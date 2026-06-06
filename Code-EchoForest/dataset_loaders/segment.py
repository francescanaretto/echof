from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


DATASET_NAME = "segment"
OUT_DIR = "../Data-original/segment"
TEST_SIZE = 0.3
RANDOM_STATE = 42


def main():
    ds = load_dataset("mstz/segment", "segment")
    df = ds["train"].to_pandas()
    print(df)

    y = df.pop("class")

    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    x = pre.fit_transform(df).astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y)
    print("X shape:", x.shape, "  y classes:", dict(zip(le.classes_, range(len(le.classes_)))))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pd.DataFrame(x_train).to_csv(f"{OUT_DIR}/train_set_{DATASET_NAME}.csv", index=False)
    pd.DataFrame(x_test).to_csv(f"{OUT_DIR}/test_set_{DATASET_NAME}.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{OUT_DIR}/train_labels_{DATASET_NAME}.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{OUT_DIR}/test_labels_{DATASET_NAME}.csv", index=False)


if __name__ == "__main__":
    main()
