from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from datasets import load_dataset

dataset = load_dataset("mstz/isolet", "isolet")["train"].to_pandas()
print(dataset.columns)
# Separate the label column.
y = dataset.pop("617")   # adjust if the label column has a different name

# 3) Identify numeric vs non-numeric columns
cat_cols = dataset.select_dtypes(include=["object", "category", "string"]).columns.tolist()
num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

# 4) Column transformer: numeric passthrough, categorical one-hot
pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop"
)

X = pre.fit_transform(dataset).astype(np.float32)

# 5) Encode labels as integers
le = LabelEncoder()
y = le.fit_transform(y)
print("X shape:", X.shape, "  y classes:", dict(zip(le.classes_, range(len(le.classes_)))))

#y = df.pop("class").to_numpy()
#X = df.to_numpy(dtype=np.float32)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)
pd.DataFrame(X_train).to_csv('../Data-original/glass/train_set_glass.csv', index = False)
pd.DataFrame(X_test).to_csv('../Data-original/glass/test_set_glass.csv', index = False)
pd.DataFrame(Y_train).to_csv('../Data-original/glass/train_labels_glass.csv', index = False)
pd.DataFrame(Y_test).to_csv('../Data-original/glass/test_labels_glass.csv', index = False)
