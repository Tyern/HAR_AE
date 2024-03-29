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
        A_array = np.abs(np.random.randn(7, 10000, 6))

        # data = np.empty(0,)
        for array_2 in A_array:
            T = 4 #周期(sample数)
            for array_1 in array_2:
                phase = np.random.randint(0, 359)
                data_array = np.empty(0)
                for a in array_1:
                    print(phase)
                    list = []
                    for i in range(500):
                        list.append((360/T)*i % 360)
                    list = np.array(list)
                    y = a*np.round(np.sin(np.radians(list + phase)),8)
                    print(list)
                    plt.plot(y)
                    plt.show()
                    data_array = np.stack(data_array, y)
                    print(data_array.shape)
        
        print('-----------------------------------------------')

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
