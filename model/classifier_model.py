import os

import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.optim import Optimizer

from model.base_model import ClassifierBaseModel

class Classifier1D(ClassifierBaseModel):
    def __init__(
            self, 
            optimizer: Optimizer=None, 
            optimizer_param: dict=None,
            cnn_channel_param = [
                (6, 32, 8, 0, 3),
                (32, 64, 8, 0, 3)
            ],
            linear_channel_param = [
                1024, 256, 128
            ],
            out_class_num = 8,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)

        cnn_list = []
        for cnn_channel in cnn_channel_param:
            cnn_list.extend([
                nn.Conv1d(
                    in_channels=cnn_channel[0], 
                    out_channels=cnn_channel[1],
                    kernel_size=cnn_channel[2], 
                    padding=cnn_channel[3], 
                    stride=cnn_channel[4]
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ])
        self.cnn = nn.Sequential(*cnn_list)

        self.example_temp_out = self.cnn(self.example_input_array)
        lin_in_features = self.example_temp_out.shape[1:].numel()

        linear_list = []
        for linear_channel_idx in range(len(linear_channel_param)):

            lin_out_features = linear_channel_param[linear_channel_idx]
            linear_list.extend([
                nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                nn.BatchNorm1d(lin_out_features),
                nn.ReLU(),
            ])
            lin_in_features = lin_out_features

        linear_list.append(nn.Linear(in_features=lin_in_features, out_features=out_class_num))
        self.linear = nn.Sequential(*linear_list)

    def forward(self, x):
        out = self.cnn(x)
        out = self.linear(out.view(out.shape[0], -1))
        return out
    

class Classifier1DBNModel(ClassifierBaseModel):
    def __init__(
            self, 
            optimizer: Optimizer=None, 
            optimizer_param: dict=None,
            cnn_channel_param = [
                (6, 32, 8, 0, 3),
                (32, 64, 8, 0, 3)
            ],
            linear_channel_param = [
                1024, 256, 128
            ],
            out_class_num = 8,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)

        cnn_list = []
        for cnn_channel in cnn_channel_param:
            cnn_list.extend([
                nn.Conv1d(
                    in_channels=cnn_channel[0], 
                    out_channels=cnn_channel[1],
                    kernel_size=cnn_channel[2], 
                    padding=cnn_channel[3], 
                    stride=cnn_channel[4]
                ),
                nn.BatchNorm1d(num_features=cnn_channel[1]),
                nn.ReLU(),
            ])
        self.cnn = nn.Sequential(*cnn_list)

        self.example_temp_out = self.cnn(self.example_input_array)
        lin_in_features = self.example_temp_out.shape[1:].numel()

        linear_list = []
        for linear_channel_idx in range(len(linear_channel_param)):

            lin_out_features = linear_channel_param[linear_channel_idx]
            linear_list.extend([
                nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                nn.BatchNorm1d(lin_out_features),
                nn.ReLU(),
            ])
            lin_in_features = lin_out_features

        linear_list.append(nn.Linear(in_features=lin_in_features, out_features=out_class_num))
        self.linear = nn.Sequential(*linear_list)

    def forward(self, x):
        out = self.cnn(x)
        out = self.linear(out.view(out.shape[0], -1))
        return out
        

class Classifier1DRaw(ClassifierBaseModel):
    def __init__(
            self, 
            optimizer: Optimizer=None, 
            optimizer_param: dict=None,
            cnn_channel_param = [
                (6, 32, 8, 0, 3),
                (32, 64, 8, 0, 3)
            ],
            linear_channel_param = [
                1024, 256, 128
            ] 
        ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 500)

        cnn_list = []
        for cnn_channel in cnn_channel_param:
            cnn_list.extend([
                nn.Conv1d(
                    in_channels=cnn_channel[0], 
                    out_channels=cnn_channel[1],
                    kernel_size=cnn_channel[2], 
                    padding=cnn_channel[3], 
                    stride=cnn_channel[4]
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ])
        self.cnn = nn.Sequential(*cnn_list)

        self.example_temp_out = self.cnn(self.example_input_array)
        lin_in_features = self.example_temp_out.shape[1:].numel()

        linear_list = []
        for linear_channel_idx in range(len(linear_channel_param)):

            lin_out_features = linear_channel_param[linear_channel_idx]
            linear_list.extend([
                nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                nn.BatchNorm1d(lin_out_features),
                nn.ReLU(),
            ])
            lin_in_features = lin_out_features

        linear_list.append(nn.Linear(in_features=lin_in_features, out_features=8))
        self.linear = nn.Sequential(*linear_list)

    def forward(self, x):
        out = self.cnn(x)
        out = self.linear(out.view(out.shape[0], -1))
        return out
    
