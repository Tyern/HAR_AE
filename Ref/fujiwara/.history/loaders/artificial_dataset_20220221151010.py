import os
from re import S
import sys
import pickle
from tkinter import Y
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
        j=0
        data = []
        for array_1 in A_array:
            if count % 10000 == 0 :
                T = T * 2
            phase = np.random.randint(0, 359)
            data_array = np.empty((0,500))
            i=0
            y_list =[]
            list = []
            for i in range(500):
                list.append((360/T)*i % 360)
                list = np.array(list)
            for a in array_1:
                y = a*np.round(np.sin(np.radians(list + phase)),8)
                y = y.tolist()
                # plt.plot(y)
                # plt.show()
                # y = y.reshape(1, 500)
                # if i == 0 :
                #     data_array = y
                # else:
                #     data_array = np.concatenate([data_array, y], axis = 0)
                # i += 1
                y_list.append(y)
                
            # data_array = np.reshape(data_array, (1, 6, 500))
            # if j == 0 :
            #     data = data_array
            #     j += 1 
            # else:
            #     data = np.concatenate([data, data_array], axis = 0)
            data.append(y_list)
            print(np.shape(data))
            targets.append(T)
            count += 1
            print(count)
            print(T)
        targets = np.array(targets)
        np.save('datasets/artificial', data)
        np.save('datasets/artificial', targets)

        
        print('-----------------------------------------------')

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
