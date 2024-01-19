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

from data_module.data_module import FFTDataModule
from model.classifier_model import Classifier1D


# In[ ]:


TEST = False

random_seed = 42
acc_gyr_dataset_path = "dataset/processed_data_acc_gyr"
lin_gyr_dataset_path = "dataset/processed_data_lin_gyr"


# In[8]:


from lightning.pytorch.utilities.model_summary import ModelSummary

net = Classifier1D(
    optimizer=optim.SGD,
    optimizer_param={
        "learning_rate": 0.01,
        "momentum": 0.5,
    }, 
    cnn_channel_param = [
        (6, 32, 8, 0, 3),
        (32, 64, 8, 0, 3)
    ],
    linear_channel_param = [
        256, 128
    ]).to("cpu")

model_summary = ModelSummary(net, max_depth=6)
print(model_summary)


# In[9]:


n_epochs = 20
patience = 1000
optimizer_param_dict = {
    "Adam": (optim.Adam, {
        "lr": 0.001,
    }),
    "SGD": (optim.SGD, {
        "lr": 0.001,
        "momentum": 0.5,
    }),
}
optimizer, optimizer_param = optimizer_param_dict["Adam"]
dataset_path=lin_gyr_dataset_path

log_save_dir = "lightning_logs"
log_save_name = "08_classify"


# In[ ]:


print(" ----------------------start training---------------------------")
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

tensorboard_logger = TensorBoardLogger(save_dir=log_save_dir, name=log_save_name,)
csv_logger = CSVLogger(save_dir=log_save_dir, name=log_save_name,)
checkpoint_callback = ModelCheckpoint(
    dirpath=None,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="sample_{epoch:02d}-{step:02d}-{val_loss:02f}"
)

trainer = L.Trainer(
    logger=[tensorboard_logger, csv_logger],
    callbacks=[EarlyStopping(monitor="val_loss", patience=patience), checkpoint_callback],
    max_epochs=n_epochs,
    check_val_every_n_epoch=10,
    )

net = Classifier1D(
    optimizer = optimizer,
    optimizer_param = optimizer_param, 
    cnn_channel_param = [
        (6, 32, 8, 0, 3),
        (32, 64, 8, 0, 3)
    ],
    linear_channel_param = [
        256, 128
    ]
)

data_module = FFTDataModule(dataset_path=lin_gyr_dataset_path, batch_size=8192)

trainer.fit(model=net, datamodule=data_module)
trainer.test(model=net, datamodule=data_module)

