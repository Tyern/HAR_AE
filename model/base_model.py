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

class AEBaseModel(L.LightningModule):
    def __init__(self):
        super().__init__()

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

    def predict_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        return output


class ClassifierBaseModel(L.LightningModule):
    def __init__(self):
        super().__init__()

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

    def predict_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)

        test_pred_label = torch.argmax(output, dim=1)
        return test_pred_label
        
