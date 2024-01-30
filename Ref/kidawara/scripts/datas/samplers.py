#! -*- coding: utf-8
import torch
import numpy as np
import random
from datas import *
from collections import OrderedDict
import pandas as pd
import os.path as path

#__all__ = []
__all__ = ["UnsertaintySampler", "RandSampler", "BiasSampler", "TrainedSampler", "Trained_BiasSampler"]

class UnsertaintySampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 sample_size: int = 1024):
        """
        コンストラクタ

        データのunsertainty（不確実さ）の順にデータを抽出するsampler.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            データセット
        sampling_size: int, default=1024
            サンプリングサイズ。
            0の場合、データセット内の全データを対象とする。
            0より大きい値を指定した場合、unsertaintyの大きい順に指定件数までを対象とする。

        Returns
        ----------
            UnsertaintySamplerのインスタンスを返す
        """
        super(UnsertaintySampler, self).__init__(dataset)
        self.__dataset__ = dataset
        self.__sample_size__ = sample_size
        self.__unsertainties__ = torch.Tensor(self.sample_size)
        torch.nn.init.zeros_(self.__unsertainties__)
        # unsertaintiesを降順に並び替えて、そのインデックスを取得
        self.__indices__ = self.unsertainties.argsort(dim=-1, descending=False)

    @property
    def dataset(self): return self.__dataset__
    @property
    def sample_size(self):
        if self.__sample_size__ > 0:
            return self.__sample_size__
        else:
            return len(self.dataset)

    @property
    def unsertainties(self): return self.__unsertainties__
    @unsertainties.setter
    def unsertainties(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        assert len(value) == len(self.dataset), "%s vs %s" % (len(value), len(self.dataset))
        self.__unsertainties__ = value
        self.__indices__ = self.unsertainties.argsort(dim=-1, descending=False)
        #self.weight = 1/(np.abs(np.random.normal(0, 1, self.sample_size))+1e-7)
        #self.weight = np.sort(self.weight)[::-1]/sum(self.weight)

    def __len__(self): return self.sample_size

    def __iter__(self):
        return iter(self.__indices__.tolist()[:self.sample_size])
        #indices = torch.cat(self.__indices__[:self.__sample_size__//2], self.__indices__[-self.__sample_size__//2:],dim=0)
        #return iter(indices)
        #return iter(np.random.choice(self.__indices__.tolist(), size = self.sample_size,p= self.weight, replace=False))

class RandSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 sample_size = 1024,
                 ):

        super(RandSampler, self).__init__(dataset)
        self._datasize = len(dataset)
        self._sample_size = sample_size
        self._dataset = dataset
        self._randomsamps = dataset

    @property
    def sample_size(self):
        return self._sample_size
    @sample_size.setter
    def sample_size(self, sample_size):
        if sample_size > 0:
            self._sample_size =sample_size
        else:
            self._sample_size =len(self._dataset)

    @property
    def randomsamps(self):
        return self._randomsamps
    @randomsamps.setter
    def randomsamps(self, sample_size):
        self.sample_size = sample_size
        self._indices = random.sample(range(self._datasize), k=self.sample_size)
        #self._indices =range(self._datasize)
        #突貫工事のゾーン
        """
        idx_car_1 = torch.where(self._dataset.car_ids == 11)[0].tolist()
        self._indices =random.sample((idx_car_1), k=self.sample_size//4)
        idx_car_5 = torch.where(self._dataset.car_ids == 1)[0].tolist()
        self._indices +=random.sample((idx_car_5), k=self.sample_size//4)
        idx_car_10 = torch.where(self._dataset.car_ids == 5)[0].tolist()
        self._indices +=random.sample((idx_car_10), k=self.sample_size//4)
        idx_car_11 = torch.where(self._dataset.car_ids == 11)[0].tolist()
        self._indices +=random.sample((idx_car_11), k=self.sample_size//4)
        """

    def __len__(self):
        return self.sample_size

    def __iter__(self):
        return iter(self._indices)

class BiasSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 sample_size: int =1024,
                 target_event: str = "DOOR",
                 event_order = OrderedDict([("ROOL",0), ("RUN", 1), ("DOOR", 2)])
                 ):
        super(BiasSampler, self).__init__(dataset)
        self._dataset = dataset
        self._target_event = target_event
        self._sample_size = sample_size
        self._event_order = event_order


    @property
    def target_event(self):
        return self._target_event

    def target_event(self, target_event):
        self._target_event=target_event

    """@property
    def biassamps(self):
        return self._biassamps
    @biassamps.setter"""
    def biassamps(self, target_ratio=1.):
        Event_True = (self._dataset.events[:,self._event_order[self._target_event]]>self._dataset.thresholds[self._target_event])*1
        Event_False = 1- Event_True
        self.Event_Tidx = np.nonzero(Event_True.numpy())
        self.Event_Fidx = np.nonzero(Event_False.numpy())

        self.target_size = int(self._sample_size*target_ratio)
        if self.target_size <= len(self.Event_Tidx[0]):
            non_target_size = self._sample_size-self.target_size
        else:
            self.target_size=len(self.Event_Tidx[0])
            non_target_size = self._sample_size-self.target_size
        self._indices = random.sample(self.Event_Tidx[0].tolist(), k=self.target_size)+random.sample(self.Event_Fidx[0].tolist(), k=non_target_size)

    def __len__(self):
        return self._sample_size

    def __iter__(self):
        return iter(self._indices)

class TrainedSampler(torch.utils.data.Sampler):
    def __init__(self,
                 trained_dataset: torch.utils.data.Dataset,
                 additional_dataset: torch.utils.data.Dataset,
                 datafile :str = "train.csv",
                 datakey :str = "original",
                 sample_size :int = 100,
                 outdir_initial_train="results\\resultsfix",
                 **kwargs
                 ):


        self._trained_dataset = trained_dataset
        self._add_dataset = additional_dataset
        self._dataset = CatDataset(self._trained_dataset, self._add_dataset)
        super(TrainedSampler, self).__init__(self._dataset)
        self._trained_datafile = path.join(outdir_initial_train, datafile)
        self._datakey =datakey
        self.trained_data = pd.read_csv(self._trained_datafile)
        self._trained_idx =self.trained_data.loc[:,self._datakey].values
        #self._sample_size = sample_size
        self._add_size = sample_size



        self._unsertainties = torch.Tensor(len(self._add_dataset))
        torch.nn.init.zeros_(self._unsertainties)
        # unsertaintiesを降順に並び替えて、そのインデックスを取得T
        self._add_indices = self._unsertainties.argsort(dim=-1, descending=False)
        #self._indices=self._trained_idx.tolist()+self._add_indices.tolist()

    @property
    def dataset(self): return self._dataset
    @property
    def sample_size(self):
        if self._add_size > 0:
            return self._add_size
        else:
            return len(self._add_dataset)

    @property
    def unsertainties(self): return self._unsertainties
    @unsertainties.setter
    def unsertainties(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        #assert len(value) == len(self._add_dataset), "%s vs %s" % (len(value), len(self.__init__add_dataset))
        ##uncertaintyを使うのはこちら（こちらを使用する場合#**の行をコメントアウトしてください。）
        self._unsertainties = value
        self._add_indices = (self._unsertainties.argsort(dim=-1, descending=False)[:self._add_size]+len(self._trained_dataset)).tolist()
        ##
        ####positiveの例からランダムに抽出するのはこちら
        #self._unsertainties = value.numpy() #**
        #self._add_indices = random.sample((((self._unsertainties>=0)*1).nonzero()[0]+len(self._trained_dataset)).tolist(), k=self._add_size) #**
        """try:
            self._add_indices = random.sample((((self._unsertainties>=0)*1).nonzero()[0]+len(self._trained_dataset)).tolist(), k=self._add_size) #**
        except:
            self._add_indices =  (((self._unsertainties>=0)*1).nonzero()[0]+len(self._trained_dataset)).tolist() + random.sample((((self._unsertainties<0)*1).nonzero()[0]+len(self._trained_dataset)).tolist(), k=self._add_size-sum(self._unsertainties>=0))
        else:
            pass"""
        ###
    def __len__(self):
        return len(self._trained_idx)+self._add_size
        #return len(self._trained_idx)
    def __iter__(self):
        #return iter(self._trained_idx.tolist()+self._add_indices[:self._add_size].tolist())
        #return iter(self._trained_idx.tolist()+(self._add_indices[:self._add_size]+len(self._trained_dataset)).tolist())

        return iter(self._trained_idx.tolist()+self._add_indices)

class Trained_BiasSampler(torch.utils.data.Sampler):
    #先行学習データのindexと
    def __init__(self,
                 trained_dataset: torch.utils.data.Dataset,
                 additional_dataset: torch.utils.data.Dataset,
                 datafile :str = "results\\flow\\1024Base200ep\\train.csv",
                 datakey :str = "original",
                 sample_size :int = 512,
                 target_event :str = "DOOR",
                 event_order = OrderedDict([("ROOL",0), ("RUN", 1), ("DOOR", 2)]),
                 target_ratio=0
                 ):


        self._trained_dataset = trained_dataset
        self._add_dataset = additional_dataset
        self._dataset = CatDataset(self._trained_dataset, self._add_dataset)
        super(Trained_BiasSampler, self).__init__(self._dataset)
        self._trained_datafile = datafile
        self._datakey =datakey
        self.trained_data = pd.read_csv(self._trained_datafile)
        self._trained_idx =self.trained_data.loc[:,self._datakey].values
        self._sample_size = sample_size
        self._add_size = sample_size
        self._event_order = event_order
        self._target_event = target_event
        self.biassamps(target_ratio)




    @property
    def dataset(self): return self._dataset
    @property
    def sample_size(self):
        if self._add_size > 0:
            return self._add_size
        else:
            return len(self._add_dataset)



    @property
    def target_event(self):
        return self._target_event

    def target_event(self, target_event):
        self._target_event=target_event

    def biassamps(self, target_ratio=1.):

        Event_True = (self._add_dataset.events[:,self._event_order[self._target_event]]>self._dataset.thresholds[self._target_event])*1
        Event_False = 1- Event_True
        self.Event_Tidx = np.nonzero(Event_True.numpy())[0]+len(self._trained_dataset)
        self.Event_Fidx = np.nonzero(Event_False.numpy())[0]+len(self._trained_dataset)

        self.target_size = int(self._sample_size*target_ratio)
        if self.target_size <= len(self.Event_Tidx):
            non_target_size = self._sample_size-self.target_size
        else:
            self.target_size=len(self.Event_Tidx)
            non_target_size = self._sample_size-self.target_size
        self._indices = random.sample(self.Event_Tidx.tolist(), k=self.target_size)+random.sample(self.Event_Fidx.tolist(), k=non_target_size)


        #target_ratio=期待値とする場合
        #self._indices = random.sample(range(len(self._trained_dataset),  len(self._trained_dataset)+len(self._add_dataset)), k=self.sample_size)

    def __len__(self):
        return len(self._trained_idx)+self._sample_size
    def __iter__(self):
        return iter(self._trained_idx.tolist()+self._indices)
