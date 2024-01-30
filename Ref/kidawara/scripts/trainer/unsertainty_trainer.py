#! -*- coding: utf-8
import os
import os.path as path
import typing
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import roc_curve

from datas import *
from models import *

import random
from .trainer import AUCTrainer

CarIds = typing.List[int]
CriterionFactor = typing.Tuple[torch.nn.Module, float]
Criterions = typing.OrderedDict[str, CriterionFactor]


class UnsertaintyTrainer(AUCTrainer):
    def __init__(self,
                 cuda: bool = False):
        super(UnsertaintyTrainer, self).__init__(cuda=cuda)

    def DataLoader(self, ncfile: str,
                   car_ids: CarIds = None,
                   sample_size: int = 1024,
                   batch_size: int = 1024,
                   data_size_per_car: int = 0,
                   shuffle: bool = False,
                   train :bool =False,
                   initdir:str=None,
                   target_ratio=0,
                   **kwargs):
        assert path.isfile(ncfile)
        assert batch_size > 0
        assert car_ids is None or len(car_ids) > 0
        #dataset = SensorDataset(ncfile,
        #                        car_ids=car_ids,
        #                        data_size_per_car=data_size)
        data_size=data_size_per_car
        if train:
            __CAR_IDS__=[1,5,10,11,13]
            init_car_ids = car_ids
            add_car_ids = [car for car in __CAR_IDS__ if not car in init_car_ids]
            print(add_car_ids)
            trained_dataset = SensorDataset(ncfile,
                                            car_ids=init_car_ids,
                                            data_size_per_car=256)

            self.additional_dataset = SensorDataset(ncfile,
                                                car_ids=add_car_ids,
                                                data_size_per_car=0)

            sampler = TrainedSampler(trained_dataset,
                                    self.additional_dataset,
                                    sample_size=sample_size,
                                    outdir_initial_train=initdir,
                                    **kwargs
                                    )
            """
            sampler = Trained_BiasSampler(trained_dataset,
                                    self.additional_dataset,
                                    sample_size=sample_size,
                                    outdir_initial_train=initdir,
                                    **kwargs
                                    )
            """
            dataset = CatDataset(trained_dataset, self.additional_dataset)
            return torch.utils.data.DataLoader(dataset,
                                               sampler=sampler,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
        else:

            dataset = SensorDataset(ncfile,
                                    car_ids=car_ids,
                                    data_size_per_car=data_size)

            return torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    def set_unsertainty(self,
                        model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        unsertainty_event: str = "DOOR",
                        batch_size: int = 1024,
                        **kwargs):
        with torch.no_grad():  # 勾配不要
            # dataloaderからSensorDataset, UnsertaintySamplerのインスタンスを取得
            dataset = self.additional_dataset
            print((self.additional_dataset.events>torch.Tensor([0.5, 0.5, 1/128])).sum(dim=0))
            sampler = dataloader.sampler
            assert isinstance(dataset, SensorDataset)
            assert isinstance(sampler, TrainedSampler)
            assert isinstance(model, ConvModelLinear)
            assert unsertainty_event in model.events()
            assert batch_size > 0

            datas = dataset.datas
            niter = int(np.ceil(len(datas) / batch_size))

            # 対象イベントの値を取得
            event_vals = dataset.events[:, dataset.event_order(
                unsertainty_event)].view(-1)
            # イベントの閾値を取得
            ethreshold = dataset.threshold(unsertainty_event)
            # イベントの値と閾値からイベントが発生しているかどうかを判断する。
            events = (event_vals >= ethreshold).float().detach().cpu().numpy()
            del event_vals, ethreshold

            # DNNからの出力値を取得
            outs = []
            for i in range(niter):
                sidxs = i*batch_size
                x = datas[sidxs:sidxs+batch_size]
                y = model(x.to(self.device))
                p = y[unsertainty_event]
                outs.append(p)
            outs = torch.cat(outs, dim=0)
            outs = outs.detach().cpu().numpy()

            # 閾値を計算
            fpr, tpr, thresholds = roc_curve(events, outs, pos_label=1.0)
            #cutoffs = np.sqrt((1-tpr)**2 + fpr**2)
            #cutoffs = 0.5*(tpr+(1-fpr))

            cutoffs = 1/(0.5*(1/tpr + 1/(1-fpr)))
            #threshold = thresholds[np.argmax(cutoffs)]

            HITS=[]
            for i, th in enumerate(thresholds):
                hits = sum(1-(events- (outs>=th).ravel().astype(float))**2)
                HITS.append(int(hits))
                if i > 2.5*(1e4):
                    print(i)
                    break
            threshold=thresholds[np.argmax(HITS)]

            #unsertainty =(abs(outs-threshold)+((outs<threshold)*1e3)).ravel() #positive_UMで使用
            unsertainty = abs(outs-threshold).ravel() #UM, RMで使用
            #unsertainty = (outs-threshold).ravel() #positive_RMで使用
            print(sum(outs<threshold))

            sampler.threshold = threshold
            #unsertainty = ((outs-threshold<=0)*1).ravel()
            sampler.unsertainties = torch.as_tensor(unsertainty)

    def train(self,
              train_file: str,
              val_file: str,
              test_file: str,
              outdir: str,
              train_car_ids: CarIds = [1, 5, 10],
              val_car_ids: CarIds = [11],
              test_car_ids: CarIds = [13],
              epochs: int = 1000,
              initial_epoch: int = 1,
              initial_model: str = None,
              sample_size: int = 1024,
              data_size: int = 1024,
              update_unsertainty_interval: int = 1,
              unsertainty_event: str = "DOOR",
              seed: int = 11,
              criterion_config: OrderedDict = OrderedDict([("AUC", 1.0),
                                                           ("BCE", 1.0)]),
              **kwargs):
        if seed is not None and seed > 0:
            torch.manual_seed(seed)
            #random.seed(seed)
        assert update_unsertainty_interval > 0

        train_loader = self.DataLoader(train_file,
                                       car_ids=train_car_ids,
                                       sample_size=sample_size,
                                       shuffle=False,
                                       train=True,
                                       **kwargs)
        self.train_sampler = train_loader.sampler
        print(len(self.train_sampler))
        self.train_events = []

        self.double_events = OrderedDict([("ROLL.RUN", 0), ("ROLL.DOOR", 0), ("RUN.DOOR", 0), ("ALL", 0)])


        #print(train_sampler.__dict__)
        val_loader = self.DataLoader(val_file,
                                        car_ids=val_car_ids,
                                        shuffle=False, data_size_per_car=256,**kwargs)
        test_loader = self.DataLoader(test_file,
                                         car_ids=test_car_ids,
                                         shuffle=False, **kwargs)

        os.makedirs(outdir, exist_ok=True)
        best_model_file = path.join(outdir, "best_model.pth")
        last_model_file = path.join(outdir, "last_model.pth")
        train_log_file = path.join(outdir, "train.log")

        model = self.Model(initial_model=initial_model, **kwargs)
        model.to(self.device)
        self.set_unsertainty(model, train_loader,
                             unsertainty_event=unsertainty_event,
                             **kwargs)
        for _, events in train_loader:
            #events = events.to(self.device)
            self.train_events.append(events.detach().cpu().sum(dim=0))
            doubeve = events.detach().numpy().copy()*[3,5,7]

            self.double_events["ROLL.RUN"] = ((doubeve.sum(axis=1)==8)*1).sum()
            self.double_events["ROLL.DOOR"] = ((doubeve.sum(axis=1)==10)*1).sum()
            self.double_events["RUN.DOOR"] = ((doubeve.sum(axis=1)==12)*1).sum()
            self.double_events["ALL"] = ((doubeve.sum(axis=1)==15)*1).sum()
            print(self.double_events)

        print(self.train_events)
        #model = self.Model(initial_model=None, **kwargs)
        #model.to(self.device)
        for name, params in model.named_parameters():
            self.logger.debug("Model parameter: %s (%s)" %
                              (name, list(params.shape)))
        criterions = self.Criterions(configs=criterion_config, **kwargs)
        optimizer = self.Optimizer(model.parameters(), **kwargs)

        # initialize unsertainty


        metric_score = np.nan
        with open(train_log_file, "w") as train_log:
            for epoch in range(initial_epoch, initial_epoch + epochs):
                logs = OrderedDict([("epoch", epoch)])
                model.train()
                tag = "train"
                tmp_log = self.epoch(epoch, model, train_loader, criterions, optimizer,
                                     tag=tag, is_train=True,
                                     **kwargs)
                for k, v in tmp_log.items():
                    logs["%s_%s" % (tag, k)] = v

                # validation and test
                model.eval()
                with torch.no_grad():  # no_gradを指定して、計算グラフを作成しない
                    for tag, loader in zip(["val", "test"],
                                           [val_loader, test_loader]):
                        tmp_log = self.epoch(epoch, model, loader, criterions, optimizer,
                                             tag=tag, is_train=False,
                                             **kwargs)
                        for k, v in tmp_log.items():
                            logs["%s_%s" % (tag, k)] = v

                #self.logger.info("%06d %s" % (epoch, logs))
                if epoch == initial_epoch:  # write header.
                    train_log.write(",".join(list(logs.keys())))
                    train_log.write("\n")
                line = ",".join(["%0.6f" % v for v in logs.values()])
                train_log.write(line)
                train_log.write("\n")

                score = logs[self.metric_key]
                if np.isnan(metric_score) or (metric_score >= score
                                              if self.metric_mode == "min"
                                              else metric_score <= score):
                    torch.save(model.state_dict(), best_model_file)
                    self.logger.info("update best model: %0.6f -> %0.6f" %
                                     (metric_score, score))
                    metric_score = score

                if epoch % update_unsertainty_interval == 0:
                    self.set_unsertainty(model, train_loader,
                                         unsertainty_event=unsertainty_event,
                                         **kwargs)
                    self.logger.debug("update unsertainty")

        torch.save(model.state_dict(), last_model_file)
        if not path.isfile(best_model_file):
            torch.save(model.state_dict(), best_model_file)

        # TODO plot unsertainty
