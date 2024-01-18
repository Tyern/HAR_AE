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

from model.AE_model import AE1D_simple


class AE1DClassifier_simple(L.LightningModule):
    def __init__(self, AE1D_ckpt_path, optimizer: Optimizer=None, optimizer_param: dict=None):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)

        self.feature_extractor = AE1D_simple.load_from_checkpoint(AE1D_ckpt_path)
        self.feature_extractor.freeze()

        self.encoder = self.feature_extractor.encoder
        self.enc_linear = self.feature_extractor.enc_linear

        self.cont_classifier = nn.Sequential(
            # nn.Dropout1d(p=0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            # nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=8),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.enc_linear(out.view(out.shape[0], -1))
        out = self.cont_classifier(out)
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
        loss = F.cross_entropy(output, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        self.log("val_loss", loss, prog_bar=True)

        acc = (torch.argmax(output, dim=1) == y).sum() / len(y)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        self.log("test_loss", loss, prog_bar=False)

        acc = (torch.argmax(output, dim=1) == y).sum() / len(y)
        self.log("test_acc", acc, prog_bar=True)
