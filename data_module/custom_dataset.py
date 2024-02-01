
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, data, label, normalize=False):
        super().__init__()
        self.data = np.array(data).astype(np.float32)
        self.label = np.array(label).astype(np.int64)

        assert len(self.data) == len(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx] - 1

class DoubleDataset(Dataset):
    def __init__(self, data, label, additional_data, additional_label, normalize=False):
        super().__init__()
        self.data = data.astype(np.float32)
        self.label = label.astype(np.int64)
        self.additional_data = additional_data.astype(np.float32)
        self.additional_label = additional_label.astype(np.int64)
        
        assert normalize == False, "normalize is not implemented"
        assert len(self.data) == len(self.label)

        self.concat_data = np.concatenate([self.data, self.additional_data])
        self.concat_label = np.concatenate([self.data, self.additional_data])

    def __len__(self):
        return len(self.concat_data)

    def __getitem__(self, idx):
        return self.concat_data[idx], self.concat_label[idx] - 1