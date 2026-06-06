from bbox import AbstractBBox
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import tensor, from_numpy
from sklearn.metrics import classification_report

class MyDataset(Dataset):
    def __init__(self, xy):
        xy = np.vstack(xy).astype(np.float32)
        self.len = xy.shape[0]
        #self.x_data = from_numpy(xy[:, 0:-1]).type(torch.float32)
        self.x_data = from_numpy(xy).type(torch.float32)
        self.y_data = from_numpy(xy[:, [-1]]).type(torch.LongTensor)
        self.y_data = torch.squeeze(self.y_data)
        #print(self.x_data, self.y_data)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len


class sklearn_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier):
        super().__init__()
        self.bbox = classifier

    def model(self):
        return self.bbox

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)


class pytorch_classifier_wrapper:
    def __init__(self, net, n_classes: int, device: str = "cpu"):

        self.net = net.to(device).eval()
        self.device = device
        self.n_classes = n_classes
        self._feature_order = None

    def set_feature_order(self, cols):
        self._feature_order = list(cols)

    def predict(self, X_df):
        import numpy as np, pandas as pd, torch

        if isinstance(X_df, pd.DataFrame):
            if self._feature_order is not None:
                X_df = X_df[self._feature_order]
            X_np = X_df.to_numpy(dtype=np.float32, copy=False)  # <<< float32
        else:
            X_np = np.asarray(X_df, dtype=np.float32)

        with torch.no_grad():
            X_t = torch.from_numpy(X_np).to(self.device)
            logits = self.net(X_t)
            if logits.ndim == 1 or logits.shape[1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
                return (probs >= 0.5).cpu().numpy().astype(int)
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()
