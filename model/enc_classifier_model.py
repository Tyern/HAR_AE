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

from model.AE_model import AE1D_simple, AECNN1DModel
from model.base_model import ClassifierBaseModel

class AE1DClassifier_simple(ClassifierBaseModel):
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


class AE1DClassifier(ClassifierBaseModel):
    def __init__(self, 
            AE1D_ckpt_path, 
            linear_channel_param=[1024, 256, 128], 
            is_dropout=False,
            out_features=5,
            optimizer: Optimizer=None, 
            optimizer_param: dict=None
        ):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, 6, 257)
        self.feature_extractor = AECNN1DModel.load_from_checkpoint(AE1D_ckpt_path).to(self.device)
        self.feature_extractor.freeze()

        self.encoder = self.feature_extractor.enc_cnn
        self.enc_linear = self.feature_extractor.enc_linear

        with torch.inference_mode():
            out = self.encoder(self.example_input_array)
            self.example_temp_out = self.enc_linear(out.view(out.shape[0], -1))

        lin_in_features = self.example_temp_out.shape[1:].numel()
        # define the linear encoding sequential model
        classifier_linear_list = []
        for linear_channel_idx in range(len(linear_channel_param)):
            lin_out_features = linear_channel_param[linear_channel_idx]

            if is_dropout:
                classifier_linear_list.extend([
                    nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                    nn.BatchNorm1d(lin_out_features),
                    nn.Dropout1d(p=0.5),
                    nn.ReLU(),
                ])
            else:
                classifier_linear_list.extend([
                    nn.Linear(in_features=lin_in_features, out_features=lin_out_features),
                    nn.BatchNorm1d(lin_out_features),
                    nn.ReLU(),
                ])

            lin_in_features = lin_out_features

        classifier_linear_list.append(
            nn.Linear(in_features=lin_in_features, out_features=out_features)
        )

        self.classifier_linear = nn.Sequential(*classifier_linear_list)

    def forward(self, x):
        out = self.encoder(x)
        out = self.enc_linear(out.view(out.shape[0], -1))
        out = self.classifier_linear(out)
        return out