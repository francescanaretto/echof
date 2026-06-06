from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("mstz/seeds", "seeds", split='train')

print(ds)
ds = ds.to_pandas()
class_counts = ds['class'].value_counts()
print(class_counts)
#print(dataset.shape, dataset['class'])
labels = ds.pop('class')
X_train, X_test, Y_train, Y_test = train_test_split(ds, labels, test_size = 0.2, random_state = 42, stratify=labels)
#print(X_train.shape, X_test.shape, Y_train.shape)

ds.to_csv('../Data-original/seeds/seeds.csv')
X_train.to_csv('../Data-original/seeds/train_set_seeds.csv')
X_test.to_csv('../Data-original/seeds/test_set_seeds.csv')
Y_train.to_csv('../Data-original/seeds/train_labels_seeds.csv')
Y_test.to_csv('../Data-original/seeds/test_labels_seeds.csv')
