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
        A_array = np.abs(np.random.randn(7, 6, 10000))

        
        for array in A_array:
            for a in array:
                list = []
                T = 4 #周期(sample数)
                for i in range(500):
                    list.append((int((360/T))*i)% 360)
                print(list)
                y = a*np.round(np.sin(np.radians(list)),8)
                print(y)
                plt.plot(y)
                plt.show()
        
        print('-----------------------------------------------')
