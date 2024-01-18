import os

import numpy as np
import pandas as pd
import lightning as L

from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from data_module.custom_dataset import SimpleDataset

class FFTDataModule(L.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 1024, ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        dataset_path = self.hparams.dataset_path

        fft_train_data = np.load(os.path.join(dataset_path, "torso_train_fft.npy"))
        label_train_data = np.load(os.path.join(dataset_path, "torso_train_label.npy"))

        fft_val_data = np.load(os.path.join(dataset_path, "torso_val_fft.npy"))
        label_val_data = np.load(os.path.join(dataset_path, "torso_val_label.npy"))

        fft_test_data = np.load(os.path.join(dataset_path, "torso_test_fft.npy"))
        label_test_data = np.load(os.path.join(dataset_path, "torso_test_label.npy"))

        self.train_set = SimpleDataset(fft_train_data, label_train_data)
        self.val_set = SimpleDataset(fft_val_data, label_val_data)
        self.test_set = SimpleDataset(fft_test_data, label_test_data)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
        