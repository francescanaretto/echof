import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff

# ====== 1. Read the ARFF file ======
file_path = "heloc"  # set the local filename here

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# ====== 2. Convert byte columns to strings when needed ======
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode("utf-8")

# ====== 3. Separate features and target ======
X = df.drop("RiskPerformance", axis=1)
y = df["RiskPerformance"].astype(int)

# ====== 4. Train-test split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ====== 5. Save CSV outputs ======
X_train.to_csv("../Data-original/heloc/train_set_heloc.csv", index=False)
X_test.to_csv("../Data-original/heloc/test_set_heloc.csv", index=False)
y_train.to_csv("../Data-original/heloc/train_labels_heloc.csv", index=False)
y_test.to_csv("../Data-original/heloc/test_labels_heloc.csv", index=False)

print("Train and test files saved successfully.")
