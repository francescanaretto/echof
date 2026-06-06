import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff

file_path = "road"  # path to the ARFF file without extension

# 1) Load ARFF
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# 2) Decode bytes -> str when needed
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.decode("utf-8")

print("Columns read from the file:")
print(list(df.columns))

# 3) Target = last column (common benchmark convention)
target_column = df.columns[-1]
print(f"Selected target: {target_column}")

X = df.drop(columns=[target_column])
y = df[target_column]

# 4) Use stratify only if each class has enough examples
stratify_arg = None
if y.nunique() > 1:
    # Avoid errors when a class has very few examples.
    min_class_count = y.value_counts().min()
    if min_class_count >= 2:
        stratify_arg = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_arg
)

# 5) Save CSV (train/test completi)
train_df = X_train.copy()
train_df[target_column] = y_train.values
test_df = X_test.copy()
test_df[target_column] = y_test.values

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Optional: save separate X/y files as well
X_train.to_csv("../Data-original/road/train_set_road.csv", index=False)
X_test.to_csv("../Data-original/road/test_set_road.csv", index=False)
pd.Series(y_train, name=target_column).to_csv("../Data-original/road/train_labels_road.csv", index=False)
pd.Series(y_test, name=target_column).to_csv("../Data-original/road/test_labels_road.csv", index=False)

print("OK: saved train.csv, test.csv, X_train.csv, X_test.csv, y_train.csv, y_test.csv")

