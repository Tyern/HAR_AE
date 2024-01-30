#! -*- coding: utf-8
import json
import os
import os.path as path

import numpy as np
import torch
from netCDF4 import Dataset, num2date

from models import ConvModel
from losses import AUC
from collections import OrderedDict

if __name__ == "__main__":
    model = ConvModel()
    criterion = AUC()
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 1024
    with Dataset("datas/train.shiei.l8.i100.o000.s100.nc", "r") as nc:
        # 平均と分散を計算する
        dimensions = ["timestamp", "frequence", "timestep"]
        ntimestamp = nc.dimensions["timestamp"].size
        nfrequence = nc.dimensions["frequence"].size
        ntimestep = nc.dimensions["timestep"].size
        naxis = nc.dimensions["axis"].size
        ndatas = ntimestamp * nfrequence * ntimestep
        # calc means
        means = np.asarray([0.0] * naxis)
        for batch_no in range(int(np.ceil(len(nc["data"]) / batch_size))):
            sidx = batch_no * batch_size
            x = np.asarray(nc["data"][sidx:sidx+batch_size])
            means += x.sum(axis=0).sum(axis=0).sum(axis=0)
        means /= ndatas
        # calc sigma
        sigmas = np.asarray([0.0] * naxis)
        for batch_no in range(int(np.ceil(len(nc["data"]) / batch_size))):
            sidx = batch_no * batch_size
            x = np.asarray(nc["data"][sidx:sidx+batch_size])
            x = (x-means)**2
            sigmas += x.sum(axis=0).sum(axis=0).sum(axis=0)
        sigmas = np.sqrt(sigmas / ndatas)
        # print(means, sigmas)

        # nc["event"] # -> timestamp x 3 x 128
        event_order = OrderedDict(
            [(l, i) for i, l in enumerate(json.loads(nc["event"].label_json))])
        thresholds = OrderedDict(
            [("ROLL", 0.5), ("RUN", 0.5), ("DOOR", 1/128)])
        t = np.asarray([thresholds[event] for event in event_order.keys()])
        weights = np.hamming(128)
        weights /= np.sum(weights)

        for batch_no in range(2):
            sidx = batch_no * batch_size
            # 入力データ
            x = (np.asarray(nc["data"][sidx:sidx+batch_size]) - means)/sigmas
            x = torch.as_tensor(x).float()
            x = x.permute(0, 3, 1, 2)  # data, channel, x, y

            # 正解データ
            events = np.asarray(
                nc["event"][sidx:sidx+batch_size] * weights).sum(axis=-1)
            events = torch.as_tensor(events >= t).float()

            probs = model(x)
            logs = OrderedDict()
            update_loss = torch.zeros(1)
            for event, prob in probs.items():
                # 各イベントの情報から対象イベントの発生フラグを取得する
                e = events[:, event_order[event]].view(-1, 1)
                loss = criterion(e, prob)
                print(batch_no, event, e.sum(), loss.detach().cpu().item())
                update_loss.add_(loss)
                logs[event] = loss.detach().cpu().item()
            print(logs)
            optimizer.zero_grad()
            update_loss.backward()
            optimizer.step()
