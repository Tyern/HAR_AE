import os
from re import S
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
import matplotlib.pyplot as plt

class Artifical(Dataset):
    def __init__(self, seed):
        super().__init__()
        np.random.seed(seed)
        A_array = np.abs(np.random.randn(70000, 6))

        data = np.empty((0, 6, 500))
        targets = []

        T = 1 #周期(sample数)
        count = 0
        for array_1 in A_array:
            if count % 10000 == 0 :
                T = T * 2
            phase = np.random.randint(0, 359)
            data_array = np.empty((0,500))
            i=0
            for a in array_1:
                list = []
                for i in range(500):
                    list.append((360/T)*i % 360)
                list = np.array(list)
                y = a*np.round(np.sin(np.radians(list + phase)),8)
                # plt.plot(y)
                # plt.show()
                # y = np.reshape(y, (1, 500))
                print(y.shape)
                if i == 0 :
                    data_array = y
                else:
                    data_array = np.concatenate(data_array, y)
                i += 1
                print(data_array.shape)
            data_array = np.reshape(data_array, (1, 6, 500))
            data = np.append(data, data_array, axis = 0)
            targets.append(T)
            count += 1
            print(count)
            print(T)
        targets = np.array(targets)

        
        print('-----------------------------------------------')

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
