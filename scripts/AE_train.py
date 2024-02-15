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
from model.AE_model import AECNN1DBNModel, AE1DMaxPoolBNModel
from utils.model_utils import unwrap_model

from config.channel_param_config import channel_param_dict
from config.optimizer_param_config import optimizer_param_dict
from lightning.pytorch.utilities.model_summary import ModelSummary

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

parser.add_argument("-m", "--model_name", type=str, default="0",
                    help=f"Number of class num use to train")

parser.add_argument("-r", "--random_seed", type=int, default=42,
                    help=f"Random Seed")

if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

if args.dataset_path is None:
    args.dataset_path = f"dataset/processed_concat_data_{args.class_num}_labels"

net = AE1DMaxPoolBNModel(
    **channel_param_dict[args.model_name],).to("cpu")

model_summary = ModelSummary(net, max_depth=6)
print(model_summary)

n_epochs = 20000
patience = n_epochs//100
batch_size = 512

optimizer, optimizer_param = optimizer_param_dict["Adam"]
dataset_path = f"dataset/processed_concat_data_{args.class_num}_labels"

log_save_dir = "lightning_logs"
log_save_name = f"11.1_AE/{args.model_name}-{args.class_num}-{args.train_limit_data}-{args.random_seed}"

data_module = ALDataModule_v1(dataset_path=dataset_path, batch_size=512, prefix="torso_", postfix="_fft")
data_module.limit_and_set_train_data(data_module._train_data, data_module._train_label, limit_number=args.train_limit_data)

L.seed_everything(args.random_seed)

print(" ----------------------start training---------------------------")
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

tensorboard_logger = TensorBoardLogger(save_dir=log_save_dir, name=log_save_name,)
csv_logger = CSVLogger(save_dir=log_save_dir, name=log_save_name,)
checkpoint_callback = ModelCheckpoint(
    dirpath=None,
    save_top_k=1,
    monitor="val_mse",
    mode="min",
    filename="sample_{epoch:02d}-{step:02d}-{val_loss:02f}"
)

trainer = L.Trainer(
    logger=[tensorboard_logger, csv_logger],
    callbacks=[EarlyStopping(monitor="val_mse", patience=patience), checkpoint_callback],
    max_epochs=n_epochs,
    check_val_every_n_epoch=10,
    accelerator="gpu", 
    devices=4, 
    strategy="ddp"
    )

net = AE1DMaxPoolBNModel(
    optimizer = optimizer,
    optimizer_param = optimizer_param, 
    **channel_param_dict[args.model_name],)

data_module = ALDataModule_v1(dataset_path=dataset_path, batch_size=batch_size, prefix="torso_", postfix="_fft")
data_module.limit_and_set_train_data(data_module._train_data, data_module._train_label, limit_number=args.train_limit_data)

print("np.unique(data_module.train_label, return_counts=True)", np.unique(data_module.train_label, return_counts=True))

trainer.fit(model=net, datamodule=data_module)
trainer_test_dict = trainer.logged_metrics

trainer.test(model=net, datamodule=data_module)
trainer_test_dict.update(trainer.logged_metrics)

print("trainer.logger.log_dir", trainer.logger.log_dir)

for key in trainer_test_dict.keys():
    trainer_test_dict[key] = trainer_test_dict[key].item()

with open(os.path.join(trainer.logger.log_dir, "result.json"), "w") as f:
    json.dump(trainer_test_dict, f, indent=4)

with open(os.path.join(trainer.logger.log_dir, "argparse_params.json"), "w") as f:
    json.dump(args.__dict__, f, indent=4)