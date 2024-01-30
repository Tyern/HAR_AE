#! -*- coding: utf-8
import enum
import json
import os
import os.path as path
import typing
from collections import OrderedDict

import numpy as np
import torch
from netCDF4 import Dataset, num2date
import random

CarIds = typing.List[int]

# WType = typing.TypeVar("WType", WeightType, OrderedDict[str, WeightType])

__all__ = ["SensorDataset", "CatDataset"]


class WeightType(enum.Enum):
    SUM = 0
    HAMMIONG = 1  # use periodic=False
    HANNING = 2  # use periodic=False

    @classmethod
    def value_of(cls, value):
        for e in cls:
            if e.value == value:
                return e
        raise ValueError("Invalid Enum value: %s" % value)

    @classmethod
    def name_of(cls, name):
        name = name.upper()
        for e in cls:
            if e.name == name:
                return e
        raise ValueError("Invalid Enum name: %s" % name)


class SensorDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, car_ids=None,
                 threshods: OrderedDict = OrderedDict([("ROLL", 0.5), ("RUN", 0.5), ("DOOR", 1/128)]),
                 normalize: bool = True,
                 weight_type: WeightType = WeightType.HAMMIONG,
                 data_size_per_car: int = 0):
        super(SensorDataset, self).__init__()
        assert path.isfile(datafile)
        self.__datafile__ = datafile
        target_car_ids = car_ids
        with Dataset(datafile, "r") as nc:
            car_ids = torch.as_tensor(nc["car_id"][:])
            # print(set(car_ids.numpy()))
            datas = torch.as_tensor(nc["data"][:]).float()
            events = torch.as_tensor(nc["event"][:]).float()
            timestamps = np.array(num2date(nc["timestamp"][:],
                                           units=nc["timestamp"].units,
                                           only_use_cftime_datetimes=False,
                                           only_use_python_datetimes=True))
            event_order = OrderedDict([(l, i)
                                       for i, l in enumerate(json.loads(nc["event"].label_json))])
            if target_car_ids is not None:
                indices = None
                for car_id in target_car_ids:
                    idxs = torch.where(car_ids == car_id)[0]
                    assert len(idxs) > 0
                    if data_size_per_car > 0 and len(idxs) > data_size_per_car:
                        idxs = idxs[:data_size_per_car]

                    if indices is None:
                        indices = idxs
                    else:
                        indices = torch.cat((indices, idxs), dim=0)
                    del idxs
                car_ids = car_ids[indices]
                datas = datas[indices]
                events = events[indices]
                timestamps = timestamps[indices]
                del indices
            del target_car_ids

        # センサデータの正規化
        if normalize:
            means = datas.mean(dim=(0, 2)).unsqueeze(1)
            sigmas = datas.std(dim=(0, 2)).unsqueeze(1)
            datas -= means
            datas /= sigmas
        else:
            shape = (datas.shape[1], 1, datas.shape[3])
            means = torch.zeros(shape)
            sigmas = torch.ones(shape)
        # イベント判定
        events *= self.__init_weight_event__(weight_type,
                                             fft_length=events.shape[-1])
        events = events.sum(dim=-1)

        self.__car_ids__ = car_ids
        self.__datas__ = datas.permute(0, 3, 1, 2)
        self.__means__ = means
        self.__sigmas__ = sigmas
        self.__events__ = events
        self.__timestamps__ = timestamps
        self.__thresholds__ = threshods
        self.__event_order__ = event_order
        self.threshold_vals = torch.empty(len(event_order)).float()
        for e, i in self.__event_order__.items():
            self.threshold_vals[i] = self.threshold(e)

    @property
    def car_ids(self): return self.__car_ids__
    @property
    def datas(self): return self.__datas__
    @property
    def events(self): return self.__events__
    @property
    def timestamps(self): return self.__timestamps__
    @property
    def means(self): return self.__means__
    @property
    def sigmas(self): return self.__sigmas__
    @property
    def thresholds(self): return self.__thresholds__

    def threshold(self, event):
        return self.thresholds[event]

    def event_order(self, event):
        return self.__event_order__[event]

    def __getitem__(self, index):
        # return super().__getitem__(index)
        vals = self.datas[index]
        truths = (self.events[index] >= self.threshold_vals).float()
        return vals, truths

    def __len__(self):
        return len(self.__datas__)

    @classmethod
    def __init_weight_event__(cls, weight_type: WeightType, fft_length: int = 128):
        if weight_type == WeightType.SUM:  # FFTの全特徴量を等価とみなす
            return torch.ones(fft_length)
        elif weight_type == WeightType.HAMMIONG:
            w = torch.hamming_window(fft_length, periodic=False)
            w /= w.sum()
            return w
        elif weight_type == WeightType.HANNING:
            w = torch.hann_window(fft_length, periodic=False)
            w /= w.sum()
            return w
        else:
            raise ValueError("Unknown wegiht type: %s" % (weight_type))

class CatDataset(torch.utils.data.Dataset):
    def __init__(self,
                 trained_dataset :SensorDataset =None,
                 add_dataset :SensorDataset =None,
                 thresholds: OrderedDict = OrderedDict([("ROLL", 0.5), ("RUN", 0.5), ("DOOR", 1/128)]),
                 event_order: OrderedDict = OrderedDict([("ROLL", 0), ("RUN", 1), ("DOOR", 2)])
                 ):
        assert isinstance(trained_dataset, SensorDataset)
        assert isinstance(add_dataset, SensorDataset)
        self._trained_dataset = trained_dataset
        self._add_dataset = add_dataset
        self._event_order = event_order
        self._thresholds = thresholds
        self.threshold_vals = torch.empty(len(self._event_order)).float()
        for e, i in self._event_order.items():
            self.threshold_vals[i] = self.threshold(e)

    #torch.cat((self._trained_dataset., self._add_dataset.), dim=0)
    @property
    def car_ids(self): return torch.cat((self._trained_dataset.car_ids, self._add_dataset.car_ids), dim=0)
    @property
    def datas(self): return torch.cat((self._trained_dataset.datas, self._add_dataset.datas), dim=0)
    @property
    def events(self): return torch.cat((self._trained_dataset.events, self._add_dataset.events), dim=0)
    @property
    def timestamps(self): return np.concatenate((self._trained_dataset.timestamps, self._add_dataset.timestamps))
    #@property
    #def means(self): return torch.cat((self._trained_dataset.means, self._add_dataset.means), dim=0)
    #@property
    #def sigmas(self): return torch.cat((self._trained_dataset.sigmas, self._add_dataset.sigmas), dim=0)
    @property
    def thresholds(self): return self._trained_dataset.thresholds

    def threshold(self, event):
        return self._thresholds[event]

    def event_order(self, event):
        return self._event_order[event]

    def __getitem__(self, index):
        if index <= len(self._trained_dataset)-1:
            return self._trained_dataset[index]
        else:

            return self._add_dataset[index-len(self._trained_dataset)]
        # return super().__getitem__(index)
        #vals = self.datas[index]
        #truths = (self.events[index] >= self._threshold_vals).float()
        #return vals, truths

    def __len__(self):
        return len(self._trained_dataset)+len(self._add_dataset)
