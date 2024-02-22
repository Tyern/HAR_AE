import os

import numpy as np
import pandas as pd
import lightning as L

import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from data_module.custom_dataset import SimpleDataset, DoubleDataset
from utils.data_utils import limit_filter_data_by_class, merge_data_set
from baal.active.heuristics import heuristics


class BaseDataModule(L.LightningDataModule):
    def __init__(self, 
        batch_size: int = 1024):
        
        super().__init__()
        self.save_hyperparameters()

        self.train_data, self.train_label = [], []
        self.val_data, self.val_label = [], []
        self.test_data, self.test_label = [], []
        self.pred_data = []

    def set_train_val_test_pred_data(
        self,
        train_data = None,
        train_label = None,
        val_data = None,
        val_label = None,
        test_data = None,
        test_label = None,
        pred_data = None,
    ):

        if train_data is not None or train_label is not None:
            assert len(train_data) == len(train_label)
            self.train_data = train_data
            self.train_label = train_label
        
        if val_data is not None or val_label is not None:
            assert len(val_data) == len(val_label)
            self.val_data = val_data
            self.val_label = val_label

        if test_data is not None or test_label is not None:
            assert len(test_data) == len(test_label)
            self.test_data = test_data
            self.test_label = test_label
        
        if pred_data is not None:
            self.pred_data = pred_data

    def setup(self, stage=None):
        self.train_set = SimpleDataset(self.train_data, self.train_label)
        self.val_set = SimpleDataset(self.val_data, self.val_label)
        self.test_set = SimpleDataset(self.test_data, self.test_label)
        self.pred_set = SimpleDataset(self.pred_data, np.zeros([len(self.pred_data)]))
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.pred_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
   
class FFTDataModule(BaseDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 1024, ):
        super().__init__(batch_size)
        self.save_hyperparameters()
        
    def setup(self, *args, **kwargs):
        dataset_path = self.hparams.dataset_path

        fft_train_data = np.load(os.path.join(dataset_path, "torso_train_fft.npy"))
        label_train_data = np.load(os.path.join(dataset_path, "torso_train_label.npy"))

        fft_val_data = np.load(os.path.join(dataset_path, "torso_val_fft.npy"))
        label_val_data = np.load(os.path.join(dataset_path, "torso_val_label.npy"))

        fft_test_data = np.load(os.path.join(dataset_path, "torso_test_fft.npy"))
        label_test_data = np.load(os.path.join(dataset_path, "torso_test_label.npy"))

        self.set_train_val_test_pred_data(
            train_data = fft_train_data,
            train_label = label_train_data,
            val_data = fft_val_data,
            val_label = label_val_data,
            test_data = fft_test_data,
            test_label = label_test_data,
            pred_data = fft_test_data,
        )
        super().setup(*args, **kwargs)


class DefaultDataModule(BaseDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 1024, prefix="", postfix=""):
        super().__init__(batch_size=batch_size)
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        dataset_path = self.hparams.dataset_path

        fft_train_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "train" + self.hparams.postfix + ".npy"))
        label_train_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "train_label.npy"))

        fft_val_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "val" + self.hparams.postfix + ".npy"))
        label_val_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "val_label.npy"))

        fft_test_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "test" + self.hparams.postfix + ".npy"))
        label_test_data = np.load(os.path.join(dataset_path, self.hparams.prefix + "test_label.npy"))

        self.set_train_val_test_pred_data(
            train_data = fft_train_data,
            train_label = label_train_data,
            val_data = fft_val_data,
            val_label = label_val_data,
            test_data = fft_test_data,
            test_label = label_test_data,
            pred_data = fft_test_data,
        )
        super().setup(*args, **kwargs)


class ALDataModule_v1(BaseDataModule):
    LABEL_RANGE = range(1, 8)
    LABEL_CLASS_NAME = ["still", "walking", "run", "bike", "car", "bus", "train", "subway"]
    SAMPLE_DICT = {
        "uncertainty": heuristics.Certainty,
        "entropy": heuristics.Entropy,
        "margin": heuristics.Margin,
    }
    def __init__(self, 
                 dataset_path: str, 
                 batch_size: int = 1024, 
                 prefix="", 
                 postfix="",
                 sampler_heuristic:str=None,
                 sampler_size: int=500):
        
        super().__init__(batch_size=batch_size)
        self.save_hyperparameters()

        self.sampler_heuristic = None
        if sampler_heuristic is not None:
            assert sampler_heuristic in self.SAMPLE_DICT.keys()
            self.sampler_heuristic = self.SAMPLE_DICT[sampler_heuristic]()

        dataset_path = self.hparams.dataset_path
        train_val_test_10000_5classes_file = \
            os.path.join(dataset_path, self.hparams.prefix + "train_val_test" + self.hparams.postfix + ".npz")
        train_val_test_10000_5classes_data = np.load(train_val_test_10000_5classes_file)

        self._train_data = train_val_test_10000_5classes_data["train_data"]
        self._val_data = train_val_test_10000_5classes_data["val_data"]
        self._test_data = train_val_test_10000_5classes_data["test_data"]
        self._additional_data = train_val_test_10000_5classes_data["additional_data"]

        self._train_label = train_val_test_10000_5classes_data["train_label"]
        self._val_label = train_val_test_10000_5classes_data["val_label"]
        self._test_label = train_val_test_10000_5classes_data["test_label"]
        self._additional_label = train_val_test_10000_5classes_data["additional_label"]

    def set_only_train_data(self, train_data, train_label):
        self.set_train_val_test_pred_data(
            train_data = train_data,
            train_label = train_label,
            val_data = self._val_data,
            val_label = self._val_label,
            test_data = self._test_data,
            test_label = self._test_label,
            pred_data = self._test_data,
        )

    def set_normal_train(self):
        print("set_normal_train")

        self.set_only_train_data(self._train_data, self._train_label)

    def limit_and_set_train_data(self, data, label, limit_number=-1):
        print("limit_and_set_train_data", "limit_number=", limit_number)
        assert len(data) == len(label)
        limited_train_data, limited_train_label, self.choice_limited_list = limit_filter_data_by_class(data, label, limit_number)
        self.set_only_train_data(limited_train_data, limited_train_label)

    def reduce_one_class_number(self, data, label, target_class, target_class_data_num):
        print("reduce_one_class_number", "target_class=", target_class, "target_class_data_num=", target_class_data_num)
        assert len(data) == len(label)

        if target_class_data_num == -1:
            raise NotImplementedError()
        else:
            self.reduce_one_class_choice_list = []

            target_class_idx = np.where(label == target_class)[0]
            different_class_idx = np.where(label != target_class)[0]

            choice_idx_list = np.random.choice(
                target_class_idx, 
                min(target_class_data_num, len(target_class_idx)), 
                replace=False)
            
            self.reduce_one_class_choice_list = np.concatenate([choice_idx_list, different_class_idx])

            limited_data = data[self.reduce_one_class_choice_list]
            limited_label = label[self.reduce_one_class_choice_list]

            return limited_data, limited_label


    def set_unsertainty_set(self, data, label, net_output):
        print("set_unsertainty_set")
        assert self.sampler_heuristic is not None
        assert len(data) == len(label)

        if isinstance(net_output, list):
            output_concat = torch.concat(net_output)
        else:
            output_concat = net_output

        self.sampling_rank = self.sampler_heuristic(output_concat)

        # get the k first sampler number (least confidence)
        self._uncertainty_data = data[self.sampling_rank][:self.hparams.sampler_size]
        self._uncertainty_label = label[self.sampling_rank][:self.hparams.sampler_size]

        self._train_uncertainty_data = np.concatenate([self.train_data, self._uncertainty_data])
        self._train_uncertainty_label = np.concatenate([self.train_label, self._uncertainty_label])
        
        self.set_only_train_data(self._train_uncertainty_data, self._train_uncertainty_label)

    def set_train_concat_set(self, data, label):
        print("set_train_concat_set")
        assert len(data) == len(label)
        self._train_concat_data = np.concatenate([self._train_data, data])
        self._train_concat_label = np.concatenate([self._train_label, label])

        self.set_only_train_data(self._train_concat_data, self._train_concat_label)

    def set_random_set(self, data, label):
        print("set_random_set")
        assert len(data) == len(label)

        self.random_rank = np.arange(len(data))
        np.random.shuffle(self.random_rank)

        self._random_sample_data = data[self.random_rank][:self.hparams.sampler_size]
        self._random_sample_label = label[self.random_rank][:self.hparams.sampler_size]

        self._train_random_sample_data = np.concatenate([self.train_data, self._random_sample_data])
        self._train_random_sample_label = np.concatenate([self.train_label, self._random_sample_label])

        self.set_only_train_data(self._train_random_sample_data, self._train_random_sample_label)

    def set_train_val_test_pred_merge_data(
            self, 
            train_data=None, 
            train_label=None,
            val_data=None, 
            val_label=None, 
            test_data=None, 
            test_label=None, 
            label_merge_dict=None, 
            train_limit_number=None, 
            val_limit_number=None, 
            test_limit_number=None,
            seed=None):

        print("set_train_val_test_pred_merge_data")
        limited_train_data, limited_train_label, limited_val_data, limited_val_label, limited_test_data, limited_test_label = None, None, None, None, None, None

        if any([train_data is not None, train_label is not None, label_merge_dict is not None, train_limit_number is not None]):
            if not all([train_data is not None, train_label is not None, label_merge_dict is not None, train_limit_number is not None]):
                raise ValueError("train data missing argument")
            assert len(train_data) == len(train_label)
                
            new_train_label = merge_data_set(train_label, label_merge_dict=label_merge_dict)
            limited_train_data, limited_train_label, _ = limit_filter_data_by_class(train_data, new_train_label, train_limit_number, seed=seed)
        
        if any([val_data is not None, val_label is not None, label_merge_dict is not None, val_limit_number is not None]):
            if not all([val_data is not None, val_label is not None, label_merge_dict is not None, val_limit_number is not None]):
                raise ValueError("val data missing argument")
            assert len(val_data) == len(val_label)
                
            new_val_label = merge_data_set(val_label, label_merge_dict=label_merge_dict)
            limited_val_data, limited_val_label, _ = limit_filter_data_by_class(val_data, new_val_label, val_limit_number, seed=seed)

        if any([test_data is not None, test_label is not None, label_merge_dict is not None, test_limit_number is not None]):
            if not all([test_data is not None, test_label is not None, label_merge_dict is not None, test_limit_number is not None]):
                raise ValueError("test data missing argument")
            assert len(test_data) == len(test_label)
                
            new_test_label = merge_data_set(test_label, label_merge_dict=label_merge_dict)
            limited_test_data, limited_test_label, _ = limit_filter_data_by_class(test_data, new_test_label, test_limit_number, seed=seed)

        self.set_train_val_test_pred_data(
            train_data = limited_train_data,
            train_label = limited_train_label,
            val_data = limited_val_data,
            val_label = limited_val_label,
            test_data = limited_test_data,
            test_label = limited_test_label,
            pred_data = limited_test_data,
        )

    def limit_and_set_train_val_test_pred_data(
            self, 
            train_data=None, 
            train_label=None,
            val_data=None, 
            val_label=None, 
            test_data=None, 
            test_label=None, 
            train_limit_number=None, 
            val_limit_number=None, 
            test_limit_number=None,
            seed=None):

        print("set_train_val_test_pred_merge_data")
        limited_train_data, limited_train_label, limited_val_data, limited_val_label, limited_test_data, limited_test_label = None, None, None, None, None, None

        if any([train_data is not None, train_label is not None, train_limit_number is not None]):
            if not all([train_data is not None, train_label is not None, train_limit_number is not None]):
                raise ValueError("train data missing argument")
            assert len(train_data) == len(train_label)
                
            limited_train_data, limited_train_label, _ = limit_filter_data_by_class(train_data, train_label, train_limit_number, seed=seed)
        
        if any([val_data is not None, val_label is not None, val_limit_number is not None]):
            if not all([val_data is not None, val_label is not None, val_limit_number is not None]):
                raise ValueError("val data missing argument")
            assert len(val_data) == len(val_label)
                
            limited_val_data, limited_val_label, _ = limit_filter_data_by_class(val_data, val_label, val_limit_number, seed=seed)

        if any([test_data is not None, test_label is not None, test_limit_number is not None]):
            if not all([test_data is not None, test_label is not None, test_limit_number is not None]):
                raise ValueError("test data missing argument")
            assert len(test_data) == len(test_label)
                
            limited_test_data, limited_test_label, _ = limit_filter_data_by_class(test_data, test_label, test_limit_number, seed=seed)

        self.set_train_val_test_pred_data(
            train_data = limited_train_data,
            train_label = limited_train_label,
            val_data = limited_val_data,
            val_label = limited_val_label,
            test_data = limited_test_data,
            test_label = limited_test_label,
            pred_data = limited_test_data,
        )

