from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ds = load_dataset("mstz/waveform_noise_v1", "waveformnoiseV1_1")


df = ds["train"].to_pandas()
print(df)
# 2) Separate the label
y = df.pop("class")   # adjust if the label column has a different name

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

#y = df.pop("class").to_numpy()
#X = df.to_numpy(dtype=np.float32)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)
pd.DataFrame(X_train).to_csv('../Data-original/wave-binary2/train_set_wave-binary2.csv', index = False)
pd.DataFrame(X_test).to_csv('../Data-original/wave-binary2/test_set_wave-binary2.csv', index = False)
pd.DataFrame(Y_train).to_csv('../Data-original/wave-binary2/train_labels_wave-binary2.csv', index = False)
pd.DataFrame(Y_test).to_csv('../Data-original/wave-binary2/test_labels_wave-binary2.csv', index = False)
