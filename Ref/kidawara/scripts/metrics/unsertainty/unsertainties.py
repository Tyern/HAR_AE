#! -*- coding: utf-8
import torch
from models import ConvModel
from datas import SensorDataset
import numpy as np
from sklearn.metrics import roc_curve


def random(model: ConvModel,
           dataset: SensorDataset,
           unsertainty_event: str = "ROLL",
           batch_size: int = 1024,
           **kwargs):
    device = next(model.parameters).device
    values = torch.rand(len(dataset))
    return values


def unsertainty(model: ConvModel,
                dataset: SensorDataset,
                unsertainty_event: str = "ROLL",
                batch_size: int = 1024,
                **kwargs):
    with torch.no_grad():
        datas = dataset.data
        niter = int(np.ceil(len(data)/batch_size))

        

    pass
