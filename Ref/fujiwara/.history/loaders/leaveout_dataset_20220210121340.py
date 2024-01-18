'''Dataset Class'''
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from utils import get_shuffled_idx
from torch.utils.data import Dataset
import dask.dataframe as ddf
import dask.multiprocessing
import time

class SHLLeaveOut(Dataset):

    def __init__(self, root, l_out_class, split='training', transform=None, target_transform=None,
                 download=False):
        """
        l_out_class : a list of ints. these clases are excluded in training
        """

        path_label = 'datasets/challenge-2019-train_torso/train/Torso/Label.txt'

        path_LAcc_x = 'datasets/challenge-2019-train_torso/train/Torso/Acc_x.txt'
        path_LAcc_y = 'datasets/challenge-2019-train_torso/train/Torso/Acc_y.txt'
        path_LAcc_z = 'datasets/challenge-2019-train_torso/train/Torso/Acc_z.txt'
        path_Gyr_x = 'datasets/challenge-2019-train_torso/train/Torso/Gyr_x.txt'
        path_Gyr_y = 'datasets/challenge-2019-train_torso/train/Torso/Gyr_y.txt'
        path_Gyr_z = 'datasets/challenge-2019-train_torso/train/Torso/Gyr_z.txt'
        path_list = [path_LAcc_x,path_LAcc_y,path_LAcc_z,path_Gyr_x,path_Gyr_y,path_Gyr_z]
        
        ndarray_list = np.empty((0,196072,500),float)
        i=1
        for path in path_list:
            print('load -> ',i)
            start = time.time()
            df = ddf.read_csv(path, sep = ' ', header = None)
            df_dask = df.compute()
            data_numpy = df_dask.to_numpy()
            print(data_numpy)
            print(data_numpy.shape)
            print(round(time.time() - start,5),'秒')
            data_numpy = np.reshape(data_numpy, (1, 196072, 500))
            ndarray_list = np.append(ndarray_list,data_numpy,axis=0)
            i+=1
        print(ndarray_list.shape)
        
        

        if split == 'training' or split == 'validation':
            self.train = True  # training set or test set
        else:
            self.train = False
        self.split = split
        self.l_out_class = list(l_out_class)
        # for c in l_out_class:
        #     assert c in set(list(range(10)))
        set_out_class = set(l_out_class)

        # if download:
        #     self.download()

        # if not self._check_exists():
        #     raise RuntimeError('Dataset not found.' +
        #                        ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        #data, targets = torch.load(os.path.join(self.processed_folder, data_file))
        data = self.data
        targets = self.targets

        if split == 'training':
            data = data[:50000]
            targets = targets[:50000]
        elif split == 'validation':
            data = data[50000:]
            targets = targets[50000:]

        out_idx = torch.zeros(len(data), dtype=torch.bool)  # pytorch 1.2
        for c in l_out_class:
            out_idx = out_idx | (targets == c)

        self.data = data[~out_idx]
        self.digits = targets[~out_idx]
        self.targets = self.digits

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')
class MNISTLeaveOut(MNIST):
    """
    MNIST Dataset with some digits excluded.
    
    targets will be 1 for excluded digits (outlier) and 0 for included digits.
    
    See also the original MNIST class: 
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    """
    img_size = (28, 28)

    def __init__(self, root, l_out_class, split='training', transform=None, target_transform=None,
                 download=False):
        """
        l_out_class : a list of ints. these clases are excluded in training
        """
        super(MNISTLeaveOut, self).__init__(root, transform=transform,
                                            target_transform=target_transform, download=download)
        if split == 'training' or split == 'validation':
            self.train = True  # training set or test set
        else:
            self.train = False
        self.split = split
        self.l_out_class = list(l_out_class)
        for c in l_out_class:
            assert c in set(list(range(10)))
        set_out_class = set(l_out_class)

        # if download:
        #     self.download()

        # if not self._check_exists():
        #     raise RuntimeError('Dataset not found.' +
        #                        ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        #data, targets = torch.load(os.path.join(self.processed_folder, data_file))
        data = self.data
        targets = self.targets

        if split == 'training':
            data = data[:50000]
            print(data.shape)

            targets = targets[:50000]
            print(data.shape)
        elif split == 'validation':
            data = data[50000:]
            targets = targets[50000:]

        out_idx = torch.zeros(len(data), dtype=torch.bool)  # pytorch 1.2
        for c in l_out_class:
            out_idx = out_idx | (targets == c)

        self.data = data[~out_idx]
        self.digits = targets[~out_idx]
        self.targets = self.digits

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')


class CIFAR10LeaveOut(CIFAR10):
    def __init__(self, root, l_out_class, split='training', transform=None, target_transform=None,
                 download=True, seed=1):

        super(CIFAR10LeaveOut, self).__init__(root, transform=transform,
                                          target_transform=target_transform, download=True)
        assert split in ('training', 'validation', 'evaluation')

        if split == 'training' or split == 'validation':
            self.train = True
            shuffle_idx = get_shuffled_idx(50000, seed)
        else:
            self.train = False
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)

        if split == 'training':
            self.data = self.data[shuffle_idx][:45000]
            print(self.data.shape)
            self.targets = self.targets[shuffle_idx][:45000]
            print(self.target.shape)
        elif split == 'validation':
            self.data = self.data[shuffle_idx][45000:]
            self.targets = self.targets[shuffle_idx][45000:]

        out_idx = torch.zeros(len(self.data), dtype=torch.bool)

        for c in l_out_class:
            out_idx = out_idx | (self.targets == c)
        out_idx = out_idx.bool()

        self.data = self.data[~out_idx]
        self.targets = self.targets[~out_idx]
        self._load_meta()
