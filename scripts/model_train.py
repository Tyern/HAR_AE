import os
import json
import sys

import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_module import ALDataModule_v1
from model import Classifier1D, Classifier1DBNModel, Classifier1DRaw, 
from config.channel_param_config import cnn_channel_param_dict, linear_channel_param_dict
from config.optimizer_param_config import optimizer_param_dict


random_seed = 42
L.seed_everything(random_seed)

from lightning.pytorch.utilities.model_summary import ModelSummary

import argparse

parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-ds", "--dataset_path", type=str, default=None,
                    help=f"Number of class num use to train")

parser.add_argument("-tl", "--train_limit_data", type=int, default=-1,
                    help=f"Used for limit the number of data in train data, -1 mean no limit")

parser.add_argument("-cn", "--class_num", type=int, default=6,
                    help=f"Number of class num use to train")

parser.add_argument("-cnn", "--cnn_channel_param", type=str, default="0",
                    help=f"Number of class num use to train")

if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

if args.dataset_path is None:
    args.dataset_path = f"dataset/processed_concat_data_{args.class_num}_labels"

