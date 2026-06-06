from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


dataset = load_dataset("mstz/mushroom")["train"]
print(dataset[0], dataset[1])
df = dataset.to_pandas()
print(df)
# 2) Separate the label
y = df.pop("is_poisonous")   # adjust if the label column has a different name

# 3) Identify numeric vs non-numeric columns
cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 4) Column transformer: numeric passthrough, categorical one-hot
pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop"
)

X = pre.fit_transform(df).astype(np.float32)

# 5) Encode labels as integers
le = LabelEncoder()
y = le.fit_transform(y)
print("X shape:", X.shape, "  y classes:", dict(zip(le.classes_, range(len(le.classes_)))))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)
pd.DataFrame(X_train).to_csv('../Data-original/mushroom/train_set_mushroom.csv', index = False)
pd.DataFrame(X_test).to_csv('../Data-original/mushroom/test_set_mushroom.csv', index = False)
pd.DataFrame(Y_train).to_csv('../Data-original/mushroom/train_labels_mushroom.csv', index = False)
pd.DataFrame(Y_test).to_csv('../Data-original/mushroom/test_labels_mushroom.csv', index = False)
