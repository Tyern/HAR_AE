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

from model.base_model import AEBaseModel

class AE1D_simple(AEBaseModel):
    def __init__(self, optimizer: Optimizer=None, optimizer_param: dict=None):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=8, padding=0, stride=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, padding=0, stride=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.enc_linear = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
        )

        self.dec_linear = nn.Sequential(
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64 * 26),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=8, padding=0, stride=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=6, kernel_size=8, padding=0, stride=3),
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.enc_linear(out.view(out.shape[0], -1))
        out = self.dec_linear(out)
        out = self.decoder(out.view(out.shape[0], 64, 26))
        return out


class AECNN1DModel(AEBaseModel):
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
        ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)
        
        enc_cnn_list = []
        for cnn_channel in cnn_channel_param:
            enc_cnn_list.extend([
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
        self.enc_cnn = nn.Sequential(*enc_cnn_list)

        self.example_temp_out = self.enc_cnn(self.example_input_array)
        lin_in_features = self.example_temp_out.shape[1:].numel()

        enc_linear_list = []
        for linear_channel_idx in range(len(linear_channel_param)):

            lin_out_features = linear_channel_param[linear_channel_idx]
            enc_linear_list.extend([
                nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                nn.ReLU(),
            ])
            lin_in_features = lin_out_features

        self.enc_linear = nn.Sequential(*enc_linear_list)

        dec_linear_list = []
        for linear_channel_idx in range(len(linear_channel_param) - 2, -1, -1):

            lin_out_features = linear_channel_param[linear_channel_idx]
            dec_linear_list.extend([
                nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                nn.ReLU(),
            ])
            lin_in_features = lin_out_features

        lin_out_features = self.example_temp_out.shape[1:].numel()
        dec_linear_list.append(
            nn.Linear(in_features=lin_in_features, out_features=lin_out_features)
        )
        
        self.dec_linear = nn.Sequential(*dec_linear_list)
        
        dec_cnn_list = []
        for cnn_channel_idx in range(len(linear_channel_param) - 1, -1, -1):
            cnn_channel = cnn_channel_param[cnn_channel_idx]

            dec_cnn_list.extend([
                nn.ConvTranspose1d(
                    in_channels=cnn_channel[1], 
                    out_channels=cnn_channel[0],
                    kernel_size=cnn_channel[2], 
                    padding=cnn_channel[3], 
                    stride=cnn_channel[4],
                    output_padding=1
                ),
                nn.ReLU(),
            ])

        self.dec_cnn = nn.Sequential(*dec_cnn_list)

    def forward(self, x):
        out = self.enc_cnn(x)
        out = self.enc_linear(out.view(out.shape[0], -1))
        out = self.dec_linear(out)
        out = self.dec_cnn(out.view(out.shape[0], *self.example_temp_out.shape[1:]))
        return out