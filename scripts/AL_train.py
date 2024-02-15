import os
import json
import sys
import glob

import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_module.data_module import ALDataModule_v1, FFTDataModule
from utils.data_utils import limit_filter_data_by_class
from utils.model_utils import reset_weight_model

from model.classifier_model import Classifier1DMaxPoolBNModel, Classifier1D
from utils.model_utils import unwrap_model

from config.optimizer_param_config import optimizer_param_dict
from config.channel_param_config import channel_param_dict

import argparse

parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-ck', "--checkpoint_path", type=str, default="/nfs/ksdata/tran/HAR_AE/lightning_logs/10.1_classify/4cnn_64-8-1000-42/version_0/checkpoints/sample_epoch=29-step=120-val_loss=0.565843.ckpt", 
                    help=f"AL learning model checkpoint. If you do not have any ckpt of model. You should consider running classifier notebook to create one")

parser.add_argument("--method", type=str, default="sampling",
                    help=f":: select run method from [random, sampling, full]")

parser.add_argument("-sh", "--sampling_heuristic", type=str, default="uncertainty",
                    help=f"""heuristic function in case the method is sampling:: [uncertainty, entropy, margin],
                    ref: https://github.com/baal-org/baal/blob/master/baal/active/heuristics/heuristics.py#L512""")

parser.add_argument("-ss", "--sampling_size", type=int, default=500,
                    help=f"sampler size to take from additional dataset. normally < 5000, Not effective for method `full`")

parser.add_argument("-tl", "--train_limit_data", type=int, default=1000,
                    help=f"Used for limit the number of data in train data, -1 mean no limit")

parser.add_argument("--filter_unselected_data_num", type=int, default=5000,
                    help=f"Used for get the size of additional dataset")

parser.add_argument("--reset_weight", type=int, default=0,
                    help=f"Reset weight before training. If not reset, we will continue learning from the checkpoint")

parser.add_argument("-r", "--random_seed", type=int, default=42,
                    help=f"Random Seed")

if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

L.seed_everything(42)

n_epochs = 20000
patience = n_epochs//100

optimizer, optimizer_param = optimizer_param_dict["Adam"]

log_save_dir = "lightning_logs"
reset_str = "_reset" if args.reset_weight else ""
    
log_save_name = f"16.2_AL_v3/{args.method}_{args.sampling_heuristic}_{args.sampling_size}_{args.random_seed}_{reset_str}"

net = Classifier1DMaxPoolBNModel.load_from_checkpoint(args.checkpoint_path)

model_summary = ModelSummary(net, max_depth=6)
print(model_summary)

def process_tl_and_additional_ds(data_module):
    assert args.train_limit_data + args.filter_unselected_data_num < np.max(np.unique(data_module._train_label, return_counts=True)[1])
    
    data_module.limit_and_set_train_data(data_module._train_data, data_module._train_label, limit_number=args.train_limit_data)
    print("data_module.choice_limited_list", data_module.choice_limited_list[:100])
    
    unselected_train_idx = list(set(range(len(data_module._train_label))) - set(data_module.choice_limited_list))
    unselected_train_data = data_module._train_data[unselected_train_idx]
    unselected_train_label = data_module._train_label[unselected_train_idx]

    filter_data, filter_label, choice_idx = limit_filter_data_by_class(unselected_train_data, unselected_train_label, args.filter_unselected_data_num)
    return filter_data, filter_label, choice_idx
    
assert args.method in [ "sampling",  "random",  "full"  ]

if args.method == "sampling":
    # Add to training dataset the uncertainty data for later training
    trainer = L.Trainer()
    
    data_module = ALDataModule_v1.load_from_checkpoint(args.checkpoint_path, prefix="torso_", postfix="_fft", sampler_heuristic=args.sampling_heuristic, sampler_size=args.sampling_size)
    _additional_data, _additional_label, _additional_choice_idx = process_tl_and_additional_ds(data_module)
    
    data_module.set_train_val_test_pred_data(pred_data=_additional_data)
    output = trainer.predict(model=net, datamodule=data_module)
    
    data_module.set_unsertainty_set(data=_additional_data, label=_additional_label, net_output=output)
    print("data_module.sampling_rank", data_module.sampling_rank[:100])

elif args.method == "random":
    data_module = ALDataModule_v1.load_from_checkpoint(args.checkpoint_path, prefix="torso_", postfix="_fft", sampler_size=args.sampling_size)
    _additional_data, _additional_label, _additional_choice_idx = process_tl_and_additional_ds(data_module)
    
    data_module.set_random_set(data=_additional_data, label=_additional_label)
    print("data_module.sampling_rank", data_module.random_rank[:100])
    
elif args.method == "full":
    data_module = ALDataModule_v1.load_from_checkpoint(args.checkpoint_path, prefix="torso_", postfix="_fft")
    _additional_data, _additional_label, _additional_choice_idx = process_tl_and_additional_ds(data_module)
    
    data_module.set_train_concat_set(data=_additional_data, label=_additional_label)

## Reset net weight before training
L.seed_everything(args.random_seed)
if args.reset_weight:
    reset_weight_model(net, verbose=1)

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
    accelerator="gpu", 
    devices=4, 
    strategy="ddp"
    )

trainer.fit(model=net, datamodule=data_module)
trainer_test_dict = trainer.logged_metrics

trainer.test(model=net, datamodule=data_module)
trainer_test_dict.update(trainer.logged_metrics)

for key in trainer_test_dict.keys():
    trainer_test_dict[key] = trainer_test_dict[key].item()

with open(os.path.join(trainer.logger.log_dir, "result.json"), "w") as f:
    json.dump(trainer_test_dict, f, indent=4)

with open(os.path.join(trainer.logger.log_dir, "argparse_params.json"), "w") as f:
    json.dump(args.__dict__, f, indent=4)