import os

import numpy as np
import pandas as pd
import lightning as L

from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from data_module.custom_dataset import SimpleDataset, DoubleDataset
from .sampler import build_sampler

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
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    

class DefaultDataModule(L.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 1024, prefix="", postfix=""):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        dataset_path = self.hparams.dataset_path

        fft_train_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "train" + self.hparams.postfix + ".npy"))
        label_train_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "train_label.npy"))

        fft_val_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "val" + self.hparams.postfix + ".npy"))
        label_val_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "val_label.npy"))

        fft_test_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "test" + self.hparams.postfix + ".npy"))
        label_test_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "test_label.npy"))

        self.train_set = SimpleDataset(fft_train_data, label_train_data)
        self.val_set = SimpleDataset(fft_val_data, label_val_data)
        self.test_set = SimpleDataset(fft_test_data, label_test_data)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
   
        
class ALDataModule_v1(L.LightningDataModule):
    def __init__(self, 
                 dataset_path: str, 
                 batch_size: int = 1024, 
                 prefix="", 
                 postfix="",
                 sampler_name:str=None):
        
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        dataset_path = self.hparams.dataset_path
        train_val_test_10000_5classes_file = \
            os.path.join(dataset_path, self.hparams.prefix + "train_val_test" + self.hparams.postfix + ".npz")

        train_data = train_val_test_10000_5classes_file.train_data
        val_data = train_val_test_10000_5classes_file.val_data
        test_data = train_val_test_10000_5classes_file.test_data
        additional_data = train_val_test_10000_5classes_file.additional_data

        train_label = train_val_test_10000_5classes_file.train_label
        val_label = train_val_test_10000_5classes_file.val_label
        test_label = train_val_test_10000_5classes_file.test_label
        additional_label = train_val_test_10000_5classes_file.additional_label

        self.concat_set = DoubleDataset(train_data, train_label, additional_data, additional_label)
        self.train_set = SimpleDataset(train_data, train_label)
        self.val_set = SimpleDataset(val_data, val_label)
        self.test_set = SimpleDataset(test_data, test_label)
        self.additional_set = SimpleDataset(additional_data, additional_label)
        # self.sampler = build_sampler(self.hparams.sampler_name, self.concat_set, batch_size=self.hparams.batch_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, pin_memory=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
   