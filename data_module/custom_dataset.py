
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, data, label, normalize=False):
        super().__init__()
        self.data = data.astype(np.float32)
        self.label = label.astype(np.int64)

        assert len(self.data) == len(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx] - 1