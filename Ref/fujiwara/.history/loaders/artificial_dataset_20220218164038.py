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
        A = np.random.randn(7, 6, 10000)
        print('Ashape:',A.shape)
        x = np.linspace
        y = np.sin(np.radians([0, 30, 90]))
        print(y)
        print('-----------------------------------------------')
