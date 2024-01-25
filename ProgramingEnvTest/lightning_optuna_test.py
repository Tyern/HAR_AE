#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


# In[ ]:


TEST = False

n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
L.seed_everything(random_seed)


# In[ ]:


from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class DataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        self.train_set = torchvision.datasets.MNIST(
            './files/', train=True, download=True,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
            ]))
        
        self.test_set = torchvision.datasets.MNIST(
            './files/', train=False, download=True,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
            ]))
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size_train, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size_test, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size_test, shuffle=False)


# In[ ]:


from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

class Net(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 1, 28, 28)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self.parameters(), lr=self.hparams.lr,
                      momentum=momentum)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.nll_loss(output, y)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.nll_loss(output, y)
        self.log("val_loss", loss)

        acc = (torch.argmax(output, dim=1) == y).sum()/len(y)
        self.log("val_acc", acc)

    def test_step(self,  batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        loss = F.nll_loss(output, y)
        self.log("test_loss", loss)

        acc = (torch.argmax(output, dim=1) == y).sum()/len(y)
        self.log("test_acc", acc)


# In[ ]:


def objective(trial: optuna.trial.Trial):
    lr = trial.suggest_categorical("lr", [0.001, 0.01])
    net = Net(lr=lr)
    data_module = DataModule() 

    trainer = L.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu",
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        )
    trainer.fit(model=net, datamodule=data_module)
    # trainer.test(model=net, datamodule=data_module)

    trainer_test_dict = trainer.logged_metrics
    return trainer_test_dict["val_loss"].item()


# In[ ]:


pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner())

study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=2)

print("Best trial:")
trial = study.best_trial

for key, val in trial.params.items():
    print("{}: {}".format(key, val))


# In[ ]:




