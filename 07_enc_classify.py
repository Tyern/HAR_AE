#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset


# In[2]:


TEST = False

random_seed = 42
L.seed_everything(random_seed)


# In[3]:


AE1D_checkpoint_path = "lightning_logs/AE1D/version_2/checkpoints/sample_epoch=999-step=20000-val_mse=0.009731.ckpt"


# In[4]:


fft_train_file = "dataset/processed_data/torso_train_fft.npy"
fft_val_file = "dataset/processed_data/torso_val_fft.npy"
fft_test_file = "dataset/processed_data/torso_test_fft.npy"

label_train_file = "dataset/processed_data/torso_train_label.npy"
label_val_file = "dataset/processed_data/torso_val_label.npy"
label_test_file = "dataset/processed_data/torso_test_label.npy"


# In[5]:


class CustomDataset(Dataset):
    def __init__(self, data, label, normalize=False):
        super().__init__()
        self.data = data.astype(np.float32)
        self.label = label.astype(np.int64)

        assert len(self.data) == len(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx] - 1 # Change the label's interval from 1 -> 8 to 0-> 7 


# In[6]:


from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size = 1024):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        fft_train_data = np.load(fft_train_file)
        label_train_data = np.load(label_train_file)

        fft_val_data = np.load(fft_val_file)
        label_val_data = np.load(label_val_file)

        fft_test_data = np.load(fft_test_file)
        label_test_data = np.load(label_test_file)

        self.train_set = CustomDataset(fft_train_data, label_train_data)
        self.val_set = CustomDataset(fft_val_data, label_val_data)
        self.test_set = CustomDataset(fft_test_data, label_test_data)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
        


# In[7]:


if TEST:
    data_module = DataModule()
    data_module.setup()


# In[8]:


if TEST:
    print(len(data_module.train_set))
    print(len(data_module.val_set))
    print(len(data_module.test_set))


# In[9]:


if TEST:
    val_loader = data_module.val_dataloader()
    print(next(iter(val_loader)))


# In[10]:


learning_rate = 0.01
momentum = 0.5


# In[11]:


from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

class AE1D(L.LightningModule):
    def __init__(self):
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
    


# In[12]:


if TEST:
    from lightning.pytorch.utilities.model_summary import ModelSummary

    feature_extractor = AE1D.load_from_checkpoint(AE1D_checkpoint_path)

    net = AE1D()
    model_summary = ModelSummary(net, max_depth=6)

    print(model_summary)


# In[13]:


class AE1DClassifier(L.LightningModule):
    def __init__(self, AE1D_ckpt_path):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)

        self.feature_extractor = AE1D.load_from_checkpoint(AE1D_ckpt_path)
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
        return optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum)
    
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


# In[19]:


if TEST:
    from lightning.pytorch.utilities.model_summary import ModelSummary

    net = AE1DClassifier(AE1D_ckpt_path=AE1D_checkpoint_path).to("cpu")
    model_summary = ModelSummary(net, max_depth=6)

    print(model_summary)


# In[17]:


n_epochs = 5000
patience = 50


# In[18]:


print(" ----------------------start training---------------------------")
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

tensorboard_logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name="AE1DClassifier",
)

checkpoint_callback = ModelCheckpoint(
    dirpath=None,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="sample_{epoch:02d}-{step:02d}-{val_loss:02f}"
)

trainer = L.Trainer(
    logger=tensorboard_logger,
    callbacks=[EarlyStopping(monitor="val_loss", patience=patience), checkpoint_callback],
    max_epochs=n_epochs,
    check_val_every_n_epoch=10,
    )

net = AE1DClassifier(AE1D_ckpt_path=AE1D_checkpoint_path)
data_module = DataModule(batch_size=8192)

trainer.fit(model=net, datamodule=data_module)
trainer.test(model=net, datamodule=data_module)


# In[ ]:




