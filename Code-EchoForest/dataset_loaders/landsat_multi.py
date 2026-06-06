from datasets import load_dataset
from sklearn.model_selection import train_test_split

ds = load_dataset("mstz/landsat", "landsat", split="train")
print(ds)
ds = ds.to_pandas()
class_counts = ds['class'].value_counts()
print(class_counts)
#print(dataset.shape, dataset['class'])
labels = ds.pop('class')
X_train, X_test, Y_train, Y_test = train_test_split(ds, labels, test_size = 0.2, random_state = 42, stratify=labels)
#print(X_train.shape, X_test.shape, Y_train.shape)

ds.to_csv('../Data-original/landsat-multi/landsat-multi.csv')
X_train.to_csv('../Data-original/landsat-multi/train_set_landsat-multi.csv')
X_test.to_csv('../Data-original/landsat-multi/test_set_landsat-multi.csv')
Y_train.to_csv('../Data-original/landsat-multi/train_labels_landsat-multi.csv')
Y_test.to_csv('../Data-original/landsat-multi/test_labels_landsat-multi.csv')
