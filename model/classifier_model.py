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


class Classifier1D(L.LightningModule):
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

        # self.cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=6, out_channels=32, kernel_size=8, padding=0, stride=3),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),

        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, padding=0, stride=3),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        # )

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

        # self.linear = nn.Sequential(
        #     nn.Linear(in_features=lin_in_features, out_features=1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),

        #     nn.Linear(in_features=1024, out_features=256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),

        #     nn.Linear(in_features=256, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),

        #     nn.Linear(in_features=128, out_features=8),
        # )
    
    def forward(self, x):
        out = self.cnn(x)
        out = self.linear(out.view(out.shape[0], -1))
        return out
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.hparams.optimizer is None or self.hparams.optimizer_param is None:
            raise NotImplementedError("optimizer or optimizer_param have not been set")
        
        return self.hparams.optimizer(
            self.parameters(),
            **self.hparams.optimizer_param)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, x)

        self.log("train_mse", loss, prog_bar=True)
        return loss
    
    def validation_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, x)
        self.log("val_mse", loss, prog_bar=True)

    def test_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, x)
        self.log("test_mse", loss, prog_bar=False)
    