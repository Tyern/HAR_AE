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

class AE1D_simple(L.LightningModule):
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
    