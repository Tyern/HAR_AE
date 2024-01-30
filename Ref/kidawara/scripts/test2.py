#! -*- coding: utf-8
import os
import os.path as path
from collections import OrderedDict

import torch

from datas import *
from losses import AUC
from models import ConvModel

if __name__ == "__main__":
    datafile = "datas/train.shiei.l8.i100.o000.s100.nc"
    dataset = SensorDataset(datafile, car_ids=[1, 10])
    # print(dataset.means, dataset.sigmas)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True)

    # thresholds = torch.empty(3).float()
    # for e, t in dataset.thresholds.items():
    #     i = dataset.event_order(e)
    #     thresholds[i] = t

    model = ConvModel()
    criterion = AUC()
    optimizer = torch.optim.Adam(model.parameters())

    for batch_no, (data, events) in enumerate(dataloader):
        update_loss = torch.zeros(1)
        logs = OrderedDict()

        print(data.shape, events.shape)
        print(events.sum(dim=0))

        probs = model(data)
        for event, p in probs.items():
            i = dataloader.dataset.event_order(event)
            y = events[:, i].view(-1, 1)
            loss = criterion(y, p)
            update_loss.add_(loss)
            logs[event] = loss.detach().cpu().item()
        print(batch_no, logs)
        optimizer.zero_grad()
        update_loss.backward()
        optimizer.step()
