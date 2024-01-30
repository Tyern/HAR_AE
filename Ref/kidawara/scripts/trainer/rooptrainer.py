#! -*- coding: utf-8
import os
import os.path as path
import typing
from collections import OrderedDict
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve

from datas import *
from losses import AUC
from models import ConvModelLinear

from .trainer_sigmoid import Trainer
import matplotlib.pyplot as plt
from statistics import mean, stdev
from math import sqrt

CarIds = typing.List[int]
CriterionFactor = typing.Tuple[torch.nn.Module, float]
Criterions = typing.OrderedDict[str, CriterionFactor]

__ALL__ = ["RoopTrainer"]

class RoopTrainer(Trainer):
    def __init__(self,
                 cuda: bool = False):
        super(RoopTrainer, self).__init__(cuda=cuda)

    def Model(self,
              events=["ROLL", "RUN", "DOOR"],
              common_channels=[32, 32],
              common_kernel_size=(8, 1),
              common_strides=(4, 1),
              event_channels=[64],
              event_kernel_size=(1, 8),
              event_strides=(1, 1),
              event_features=[32],
              initial_model: str = None,
              **kwargs):
        model = ConvModelLinear(events=events,
                                common_channels=common_channels,
                                common_kernel_size=common_kernel_size,
                                common_strides=common_strides,
                                event_channels=event_channels,
                                event_kernel_size=event_kernel_size,
                                event_strides=event_strides,
                                event_features=event_features).float()
        if isinstance(initial_model, str):
            initial_model = torch.load(initial_model)
        if isinstance(initial_model, torch.nn.Module):
            model.load_state_dict(initial_model)

        return model

    def epoch(self,
              epoch: int,
              model: torch.nn.Module,
              loader: torch.utils.data.DataLoader,
              criterions: Criterions,
              optimizer: torch.optim.Optimizer,
              tag: str = "",
              is_train: bool = False,
              **kwargs):
        if is_train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()
        epoch_logs = OrderedDict([("loss", 0.0), ("acc", 0.0)])

        batch_no = 0

        for datas, events in loader:

            batch_no += 1
            update_loss = torch.zeros(1).to(self.device)
            logs = OrderedDict([("loss", 0.0), ("acc", 0.0)])

            datas, events = datas.to(self.device), events.to(self.device)

            # DNNから各イベントの確率を取得する
            dnn_outs = model(datas)
            # loss計算
            for event, out in dnn_outs.items():
                i = loader.dataset.event_order(event)
                y = events[:, i].view(-1, 1)
                # モデルの出力層はLinearなのでここでSigmoidを行う。(binary cross entropy loss 対応)
                out = torch.sigmoid(out)

                for criterion_name, (criterion, factor) in criterions.items():
                    loss = criterion(out, y)

                    update_loss.add_(loss*factor)
                    logs["%s_%s_loss" %
                         (event, criterion_name)] = loss.detach().cpu().item()

            if is_train:
                optimizer.zero_grad()
                update_loss.backward()
                optimizer.step()

            logs["loss"] = update_loss.detach().cpu().item()
            for k, v in logs.items():
                if k in epoch_logs:
                    epoch_logs[k] += v
                else:
                    epoch_logs[k] = v
            self.logger.debug("%06d.%06d %5s %s" % (epoch, batch_no, tag,
                                                    ", ".join(["%s:%0.6f" % (k, v)
                                                               for k, v in logs.items()])))

        for k in epoch_logs.keys():
            epoch_logs[k] /= batch_no

        # 確率計算
        dnn_outs = OrderedDict()
        truths = OrderedDict()
        with torch.no_grad():
            for datas, events in loader:
                batch_no += 1
                update_loss = torch.zeros(1).to(self.device)

                datas = datas.to(self.device)

                # DNNから各イベントの出力値を取得する
                outs = model(datas)
                # loss計算
                for event, out in outs.items():
                    if not event in dnn_outs:
                        dnn_outs[event] = []
                        truths[event] = []

                    i = loader.dataset.event_order(event)
                    y = events[:, i].view(-1, 1)
                    dnn_outs[event].extend(out.detach().cpu().numpy().ravel())
                    truths[event].extend(y.detach().cpu().numpy().ravel())

            accs = []

            for event in model.events():
                y = np.asarray(truths[event])
                out = np.asarray(dnn_outs[event])
                fpr, tpr, thresholds = roc_curve(y, out, pos_label=1)
                auc_score = auc(fpr, tpr)
                # cutoff値計算＝accが最も良くなる閾値
                # AUC曲線をプロットした際、グラフ左上に最も近いポイントの閾値を採用する。
                """
                    # cutoffs
                    cutoffs = np.sqrt((1-tpr)**2 + fpr**2)
                    threshold = thresholds[np.argmin(cutoffs)]

                    preds = (outs >= threshold).astype(float)
                    hits = 1-(t-preds)**2
                    miss = 1-hits
                """
                # cutoffs = np.sqrt((1-tpr)**2 + fpr**2)
                # threshold = thresholds[np.argmin(cutoffs)]
                eps = 1e-7
                cutoffs = 1/(0.5 * (1/tpr+eps + 1/(1-fpr+eps)))
                threshold = thresholds[np.argmax(cutoffs)]
                preds = (out >= threshold).astype(float)
                event_acc = (1 - (y - preds)**2).mean()
                # event_acc = (1-(y-out >= threshold)**2).mean()
                epoch_logs["%s_acc" % event] = event_acc
                epoch_logs["%s_auc" % event] = auc_score
                epoch_logs["%s_threshold" % event] = threshold
                accs.append(event_acc)
            epoch_logs["acc"] = np.mean(accs)

        return epoch_logs

    def train(self,
              train_file: str,
              val_file: str,
              test_file: str,
              outdir: str,
              train_size :int,
              train_car_ids: CarIds = [1, 5, 10],
              val_car_ids: CarIds = [11],
              test_car_ids: CarIds = [13],
              epochs: int = 10,
              initial_epoch: int = 1,
              initial_model: str = None,
              seed: int = 11,
              criterion_config: OrderedDict = OrderedDict([("AUC", 1.0),
                                                           ("BCE", 1.0)]),
              **kwargs):
        sampler = RandSampler(train_file, train_size)
        """if kwargs["batch_size"] > train_size:
            kwargs["batch_size"] = 128"""
        if seed is not None and seed > 0:
            torch.manual_seed(seed)
        train_loader = self.DataLoader(train_file,
                                       sampler = sampler,
                                       car_ids=train_car_ids,
                                       sample_size = train_size,
                                       shuffle=False, **kwargs)
        self.train_events = []
        for _, events in train_loader:
            #events = events.to(self.device)
            self.train_events.append(events.detach().cpu().sum(dim=0))
        #print(train_loader[1].detach().cpu().sum(dim=0))
        #kwargs["batch_size"] = 1024
        val_loader = self.DataLoader(val_file,
                                     car_ids=val_car_ids,
                                     shuffle=False, **kwargs)
        test_loader = self.DataLoader(test_file,
                                      car_ids=test_car_ids,
                                      shuffle=False, **kwargs)
        os.makedirs(outdir, exist_ok=True)
        best_model_file = path.join(outdir, "best_model.pth")
        last_model_file = path.join(outdir, "last_model.pth")
        train_log_file = path.join(outdir, "train.log")

        model = self.Model(initial_model=initial_model, **kwargs)
        model.to(self.device)
        for name, params in model.named_parameters():
            self.logger.debug("Model parameter: %s (%s)" %
                              (name, list(params.shape)))
        criterions = self.Criterions(configs=criterion_config,
                                     **kwargs)
        optimizer = self.Optimizer(model.parameters(), **kwargs)

        metric_score = np.nan
        with open(train_log_file, "w") as train_log:
            for epoch in range(initial_epoch, initial_epoch + epochs):
                logs = OrderedDict([("epoch", epoch)])
                tag = "train"
                tmp_log = self.epoch(epoch, model, train_loader, criterions, optimizer,
                                     tag=tag, is_train=True,
                                     **kwargs)
                for k, v in tmp_log.items():
                    logs["%s_%s" % (tag, k)] = v

                # validation and test
                with torch.no_grad():  # no_gradを指定して、計算グラフを作成しない
                    for tag, loader in zip(["val"],
                                           [val_loader]):
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

        torch.save(model.state_dict(), last_model_file)
        if not path.isfile(best_model_file):
            torch.save(model.state_dict(), best_model_file)

        del initial_model

    def evaluate(self,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 outdir: str,
                 eval_size: int = 16784,
                 train_car_ids: CarIds = [1, 5, 10],
                 val_car_ids: CarIds = [11],
                 test_car_ids: CarIds = [13],
                 batch_size: int = 1024,

                 **kwargs):
        test_kwargs = kwargs.copy()
        test_kwargs["data_size"]=eval_size
        loaders = OrderedDict([
            ("train", self.DataLoader(train_file,
                                      car_ids=train_car_ids, shuffle=False, data_size = 1024,sampler=RandSampler(train_file, 1024),**kwargs)),
            ("val", self.DataLoader(val_file,
                                    car_ids=val_car_ids, shuffle=False, **kwargs)),
            ("test", self.DataLoader(test_file,
                                     car_ids=test_car_ids, shuffle=False, **test_kwargs))])

        os.makedirs(outdir, exist_ok=True)
        best_model_file = path.join(outdir, "best_model.pth")
        eval_files = OrderedDict([(tag, path.join(outdir, "evaluate.%s.csv" % tag))
                                  for tag in ["train", "val", "test"]])
        score_file = path.join(outdir, "score.csv")

        model = self.Model(**kwargs)
        model.load_state_dict(torch.load(best_model_file))
        model.eval()
        model.to(self.device)

        with torch.no_grad():
            s = OrderedDict()

            for tag, loader in loaders.items():
                dataset = loader.dataset
                eval_file = eval_files[tag]

                car_ids = dataset.car_ids
                timestamps = dataset.timestamps
                datas = dataset.datas
                events = dataset.events
                truths = events >= dataset.threshold_vals
                thresholds = dataset.thresholds
                event_order = OrderedDict([(e, dataset.event_order(e))
                                           for e in thresholds.keys()])
                evals = {"timestamps": timestamps,
                         "car_ids": ["%03d" % c for c in car_ids.detach().cpu().numpy()]}
                evals.update(
                    {"%s_like" % (e): events[:, i].detach().cpu().numpy()
                     for e, i in event_order.items()})
                evals.update(
                    {e: (events[:, event_order[e]].detach().cpu().numpy() >= t).astype(int)
                     for e, t in thresholds.items()})
                evals.update({"%s_prob" % (e): np.tile(0.0, len(datas))
                              for e in thresholds.keys()})
                evals = pd.DataFrame(evals)

                nbatch = int(np.ceil(len(datas)/batch_size))
                for batch_no in range(nbatch):
                    sidx = batch_no*batch_size
                    x = datas[sidx:sidx+batch_size].to(self.device)
                    outs = model(x)

                    for event, out in outs.items():
                        # DataFrame.locメソッドの範囲指定は＋１個出てくる
                        evals.loc[sidx: sidx+batch_size - 1,
                                  "%s_prob" % (event)] = out.detach().cpu().numpy().ravel()

                    del x, outs
                evals.to_csv(eval_file, index=False)

        with open(score_file, "wt") as f:
            headers = ["tag", "event", "n_event", "n_data",
                       "tp", "tn", "fp", "fn",
                       "accuracy", "precision", "recall", "auc", "threshold"]
            f.write(",".join(headers))
            f.write("\n")
            AUC = OrderedDict([("ROLL", 0.0), ("RUN", 0.0), ("DOOR", 0.0)])
            mic_ACC = OrderedDict([("ROLL", 0.0), ("RUN", 0.0), ("DOOR", 0.0)])
            mac_ACC = OrderedDict([("ROLL", 0.0), ("RUN", 0.0), ("DOOR", 0.0)])
            for tag, eval_file in eval_files.items():
                eval = pd.read_csv(eval_file)
                for event in model.events():
                    t = eval.loc[:, event].values
                    outs = eval.loc[:, "%s_prob" % event].values

                    fpr, tpr, thresholds = roc_curve(t, outs, pos_label=1)
                    auc_score = auc(fpr, tpr)
                    # cutoffs
                    # cutoffs = np.sqrt((1-tpr)**2 + fpr**2)
                    # threshold = thresholds[np.argmin(cutoffs)]
                    cutoffs = 1/(0.5*(1/(tpr+1e-7) + 1/(1-fpr+1e-7)))
                    threshold = thresholds[np.argmax(cutoffs)]

                    preds = (outs >= threshold).astype(float)
                    hits = 1-(t-preds)**2
                    miss = 1-hits
                    tp, tn = int(np.sum(t * hits)), int(np.sum((1-t)*hits))
                    fp, fn = int(np.sum((1-t) * miss)), int(np.sum(t * miss))
                    length = len(t)
                    t = np.sum(t)
                    mic_acc =(tp+tn)/length
                    mac_acc = 0.5*(tp/(fp+tn))+(tn/(fn+tn))
                    preds = np.sum(preds)
                    line = [tag, event, str(t), str(length),
                            str(tp), str(tn), str(fp), str(fn),
                            "%0.6f" % (np.sum(hits)/length),
                            "%0.6f" % (tp/preds),
                            "%0.6f" % (tp/t),
                            "%0.6f" % auc_score,
                            "%0.6f" % threshold]
                    if tag =="test":
                        AUC[event]=auc_score
                        mic_ACC[event]=mic_acc
                        mac_ACC[event]=mac_acc
        return AUC, mic_ACC, mac_ACC


    def plot(self, outdir, train_sizes, Scores, events, **kwargs):
        AUC={}
        mean_AUC={}
        se_AUC={}

        for event in events:
            AUC[event]= [Scores[event]["AUC"][train_size] for train_size in train_sizes]
            mean_AUC[event] = [mean(scores) for scores in AUC[event]]
            se_AUC[event] = [stdev(scores)/sqrt(len(scores)) for scores in AUC[event]]
            """mic_ACC[event]=[Scores[event]["mic_ACC"][train_size] for train_size in train_sizes]
            mean_mic_ACC[event]=[mean(scores) for scores in mic_ACC[event]]
            mac_ACC[event]=[Scores[event]["mac_ACC"][train_size] for train_size in train_sizes]
            mean_mic_ACC[event]=[mean(scores) for scores in mac_ACC[event]]"""

        plt.errorbar(train_sizes, mean_AUC["ROLL"], yerr = se_AUC["ROLL"],  ecolor='black', capsize=3, fmt='.', markersize=9, color="c", label = "ROLL")
        plt.errorbar(train_sizes, mean_AUC["RUN"], yerr = se_AUC["RUN"],  ecolor='black', capsize=3, fmt='x', markersize=7, color="m", label = "RUN")
        plt.errorbar(train_sizes, mean_AUC["DOOR"], yerr = se_AUC["DOOR"],  ecolor='black', capsize=3, fmt='^', markersize=7, color="y", label = "DOOR")
        plt.xlim([1, 1e5])
        plt.xscale("log")
        plt.xticks([1e2, 1e3, 1e4 ,1e5])
        plt.xlabel("Training Data Size", size=12)
        plt.ylabel("AUC score", size=12)
        plt.ylim([.5, 1.])
        plt.legend(loc="lower right")

        plt.show()
