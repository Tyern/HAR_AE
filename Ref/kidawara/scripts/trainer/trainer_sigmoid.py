#! -*- coding: utf-8
import os
import os.path as path
import typing
from collections import OrderedDict
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from datas import *
from losses import AUC
from models import ConvModel

CarIds = typing.List[int]
CriterionFactor = typing.Tuple[torch.nn.Module, float]
Criterions = typing.OrderedDict[str, CriterionFactor]

__all__ = ["Trainer"]


class Trainer(object):
    def __init__(self,
                 cuda: bool = False):
        super(Trainer, self).__init__()
        self.__cuda__ = cuda and torch.cuda.is_available()
        self.__events__ = ["ROLL", "RUN", "DOOR"]
        self.__metric__ = "val_loss"
        self.__logger__ = getLogger(__name__)
        self.__device__ = (torch.device("cuda")
                           if self.cuda else torch.device("cpu"))
        self.__metric_key__ = "val_acc"
        self.__metric_mode__ = "max"  # min or max

    @property
    def logger(self): return self.__logger__
    @property
    def cuda(self): return self.__cuda__
    @property
    def device(self): return self.__device__
    @property
    def events(self): return self.__events__
    @property
    def metric_key(self): return self.__metric_key__
    @property
    def metric_mode(self): return self.__metric_mode__

    def Model(self,
              events=["ROLL", "RUN", "DOOR"],
              common_channels=[32, 64],
              common_kernel_size=(8, 1),
              common_strides=(3, 1),
              event_channels=[64],
              event_kernel_size=(1, 8),
              event_strides=(1, 1),
              event_features=[128, 32],
              initial_model: str = None,
              **kwargs):
        model = ConvModel(events=events,
                          common_channels=common_channels,
                          common_kernel_size=common_kernel_size,
                          common_strides=common_strides,
                          event_channels=event_channels,
                          event_kernel_size=event_kernel_size,
                          event_strides=event_strides,
                          event_features=event_features).float()
        if isinstance(initial_model, str):
            initial_model = torch.load(initial_model)
        else:
            print("model initialized")
        if isinstance(initial_model, torch.nn.Module):
            model.load_state_dict(initial_model)

        return model




    def Criterions(self,
                   configs: OrderedDict = OrderedDict([("AUC", 1.0),
                                                       ("BCE", 1.0)]),
                   reduction="mean", **kwargs):
        criterions = OrderedDict()
        for name, factor in configs.items():
            assert factor >= 0.0, "invalid criterion factor: %f" % (factor)
            if factor == 0:
                self.logger.warn(
                    "%s criterion factor = %f, skip!" % (name, factor))
                continue
            if name == "AUC":
                criterion = AUC(reduction=reduction)
            elif name == "BCE":
                criterion = torch.nn.BCELoss(reduction=reduction)
            else:
                raise ValueError("Unsupported criterion: %s" % name)
            criterions[name] = (criterion, factor)
        assert len(criterions) > 0, "no effective criterion."
        return criterions

    def Optimizer(self, params, lr=0.001, **kwargs):
        return torch.optim.Adam(params, lr=lr)

    def DataLoader(self, ncfile: str,
                   car_ids: CarIds = None,
                   batch_size: int = 1024,
                   shuffle: bool = False,
                   sampler: torch.utils.data.Sampler = None,
                   sampler_size : int = 1024,
                   target_ratio=0,
                   data_size_per_car=0,
                   **kwargs):
        assert path.isfile(ncfile)
        assert batch_size > 0
        assert car_ids is None or len(car_ids) > 0
        dataset = SensorDataset(ncfile, car_ids=car_ids, data_size_per_car=data_size_per_car)
        #dataset = CatDataset(SensorDataset(ncfile, car_ids=car_ids), SensorDataset("datas/test.shiei.l8.i100.o000.s100.nc", car_ids=[13]))
        if sampler != None:
            if isinstance(sampler, RandSampler):
                sampler = RandSampler(dataset, sampler_size)
                sampler.randomsamps = sampler_size
            elif isinstance(sampler, BiasSampler):
                sampler = BiasSampler(dataset, sampler_size)
                sampler.biassamps(target_ratio)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler

                                           )

    def model2pdf(self, outdir: str, input_shape=(6, 65, 8), **kwargs):
        pass
        # from torchviz import make_dot
        # os.makedirs(outdir, exist_ok=True)
        # dummy = torch.rand(*[[10] + list(input_shape)])
        # model = self.Model(**kwargs)
        # out = model(dummy)
        # out = tuple(v for v in out.values())
        # graph = make_dot(out, params=dict(model.named_parameters()))
        # graph.render("model", directory=outdir, format="pdf")

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
            probs = model(datas)
            # loss計算
            for event, p in probs.items():
                i = loader.dataset.event_order(event)
                y = events[:, i].view(-1, 1)
                # print(epoch, batch_no, tag, event, y.shape, p.shape)
                # print(epoch, batch_no, tag, event, y.shape, y.detach().sum().cpu().item(),
                #       p.detach().mean().cpu().item())
                for criterion_name, (criterion, factor) in criterions.items():
                    loss = criterion(p, y)
                    # if is_train:
                    #     loss.backward(retain_graph=True)
                    # print(epoch, batch_no, tag, event, criterion_name,
                    #       type(criterion), loss, factor)

                    update_loss.add_(loss*factor)
                    logs["%s_%s_loss" %
                         (event, criterion_name)] = loss.detach().cpu().item()
                with torch.no_grad():
                    y, p = y.view(-1), p.view(-1)
                    preds = (p >= 0.5).float()  # 確率0.5以上をイベント発生とみなす
                    tptn = (1.0 - (y - preds)**2)
                    acc = tptn.sum() / len(y)
                    tp = (y * tptn).sum()
                    precision = tp / (preds.sum()+1e-8)
                    recall = tp / (y.sum()+1e-8)

                    acc = acc.detach().cpu().item()
                    logs["%s_acc" % event] = acc
                    logs["acc"] += acc
                    logs["%s_precision" %
                         event] = precision.detach().cpu().item()
                    logs["%s_recall" % event] = recall.detach().cpu().item()
                    del preds, tptn, acc, tp, precision, recall

            # self.logger.debug(loss)
            if is_train:
                optimizer.zero_grad()
                update_loss.backward()
                optimizer.step()

            logs["loss"] = update_loss.detach().cpu().item()
            logs["acc"] /= len(probs)  # イベント数で割って平均化
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

        return logs

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
              seed: int = 11,
              **kwargs):
        if seed is not None and seed > 0:
            torch.manual_seed(seed)
        train_loader = self.DataLoader(train_file,
                                       car_ids=train_car_ids,
                                       shuffle=True, **kwargs)
        val_loader = self.DataLoader(val_file,
                                     car_ids=val_car_ids,
                                     shuffle=False, **kwargs)
        test_loader = self.DataLoader(test_file,
                                      car_ids=test_car_ids,
                                      shuffle=False, **kwargs)

        os.makedirs(outdir, exist_ok=True)
        best_model_file = path.join(outdir, "best_model.pth")
        last_model_file = path.join(outdir, "last_model.pth")
        train_log_file = path.join(outdir, log)

        model = self.Model(initial_model=initial_model, **kwargs)
        model.to(self.device)
        for name, params in model.named_parameters():
            self.logger.debug("Model parameter: %s (%s)" %
                              (name, list(params.shape)))
        criterions = self.Criterions(**kwargs)
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
                    for tag, loader in zip(["val", "test"],
                                           [val_loader, test_loader]):
                        tmp_log = self.epoch(epoch, model, loader, criterions, optimizer,
                                             tag=tag, is_train=False,
                                             **kwargs)
                        for k, v in tmp_log.items():
                            logs["%s_%s" % (tag, k)] = v

                self.logger.info("%06d %s" % (epoch, logs))
                if epoch == 1:  # write header.
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

    def evaluate(self,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 outdir: str,
                 train_car_ids: CarIds = [1, 5, 10],
                 val_car_ids: CarIds = [11],
                 test_car_ids: CarIds = [13],
                 batch_size: int = 1024,
                 **kwargs):
        loaders = OrderedDict([
            ("train", self.DataLoader(train_file,
                                      car_ids=train_car_ids, shuffle=False, **kwargs)),
            ("val", self.DataLoader(val_file,
                                    car_ids=val_car_ids, shuffle=False, **kwargs)),
            ("test", self.DataLoader(test_file,
                                     car_ids=test_car_ids, shuffle=False, data_size =eval_size,  **kwargs))])

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
                    probs = model(x)

                    for event, p in probs.items():
                        # DataFrame.locメソッドの範囲指定は＋１個出てくる
                        evals.loc[sidx: sidx+batch_size - 1,
                                  "%s_prob" % (event)] = p.detach().cpu().numpy().ravel()

                    del x, probs
                evals.to_csv(eval_file, index=False)

        with open(score_file, "wt") as f:
            headers = ["tag", "event", "n_event", "n_data",
                       "tp", "tn", "fp", "fn",
                       "accuracy", "precision", "recall"]
            f.write(",".join(headers))
            f.write("\n")
            for tag, eval_file in eval_files.items():
                eval = pd.read_csv(eval_file)
                for event in model.events():
                    t = eval.loc[:, event].values
                    probs = eval.loc[:, "%s_prob" % event].values
                    preds = (probs >= 0.5).astype(float)
                    hits = 1-(t-preds)**2
                    miss = 1-hits
                    tp, tn = int(np.sum(t * hits)), int(np.sum((1-t)*hits))
                    fp, fn = int(np.sum((1-t) * miss)), int(np.sum(t * miss))
                    length = len(t)
                    t = np.sum(t)
                    preds = np.sum(preds)
                    line = [tag, event, str(t), str(length),
                            str(tp), str(tn), str(fp), str(fn),
                            "%0.6f" % (np.sum(hits)/length),
                            "%0.6f" % (tp/preds),
                            "%0.6f" % (tp/t)]
                    f.write(",".join(line))
                    f.write("\n")

    def plot(self, outdir: str, **kwargs):
        train_log_file = path.join(outdir, log)
        if not path.isfile(train_log_file):
            Warning("%s not found" % train_log_file)
            return

        if os.name == "POSIX":
            import matplotlib
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import seaborn
        import pandas as pd
        logs = pd.read_csv(train_log_file)
        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax = ax.ravel()
        tags = ["train", "val", "test"]
        img_file = path.join(outdir, "train.png")
        for tag in tags:
            for i, (valtype, a) in enumerate(zip(["acc", "loss"], ax)):
                col = "%s_%s" % (tag, valtype)
                if not col in logs.columns:
                    continue
                vals = logs[col].values
                x = list(range(1, len(vals)+1))
                a.plot(x, vals, label=tag)
                a.set_title(valtype)
                a.legend(fontsize=8)
        ax[0].set_ylim(-0.05, 1.05)
        plt.savefig(img_file)
        plt.close()

        events = ["ROLL", "RUN", "DOOR"]
        for event in events:
            fig, ax = plt.subplots(2, 3, figsize=(12, 12))
            ax = ax.ravel()
            img_file = path.join(outdir, "%s.png" % event)
            for tag in tags:
                for i, (valtype, a) in enumerate(zip(["acc", "AUC_loss", "BCE_loss",
                                                      "precision", "recall"],
                                                     ax)):
                    col = "%s_%s_%s" % (tag, event, valtype)
                    if not col in logs.columns:
                        continue
                    vals = logs[col].values
                    x = list(range(1, len(vals)+1))
                    a.plot(x, vals, label=tag)
                    if not "loss" in valtype:
                        a.set_ylim(-0.05, 1.05)
                    a.set_title(valtype.replace("_loss", ""))
                    a.legend(fontsize=8)
            ax[-1].axis("off")
            plt.suptitle(event)
            plt.savefig(img_file)
            plt.close()

        scores = pd.read_csv(path.join(outdir, "score.csv"))
        img_file = path.join(outdir, "confusion_matrix.png")
        fig, ax = plt.subplots(len(events), len(tags))
        ax = ax.ravel()
        for r, tag in enumerate(tags):
            tmp = scores[scores["tag"] == tag]
            for c, event in enumerate(events):
                i = r * len(events) + c
                a = ax[i]
                if len(tmp) > 0:
                    s = tmp[tmp["event"] == event].iloc[0]
                    x = pd.DataFrame({"T": [s.fn, s.tp], "F": [s.tn, s.fp]})
                    # x.index = ["t", "f"]
                    seaborn.heatmap(x, annot=True, cbar=False,
                                    square=True,
                                    fmt="d",
                                    yticklabels=["f", "t"],
                                    ax=a)
                a.set_ylim(0., 2.0)
                if r == len(tags)-1:
                    a.set_xlabel(event)
                if c == 0:
                    a.set_ylabel(tag)
        a = fig.add_subplot(111, frameon=False)
        a.tick_params(labeltop=False, labelbottom=False,
                      labelleft=False, labelright=False,
                      top=False, bottom=False,
                      left=False, right=False)
        a.yaxis.set_label_coords(-0.1, 0.5)
        a.xaxis.set_label_coords(0.5, -0.12)
        a.set_ylabel("predict")
        a.set_xlabel("truth")
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.13, top=0.97)
        plt.savefig(img_file)
        plt.close()

        # plot probability histgram
        bins = 100
        cmap = plt.get_cmap("tab10")
        for tag in tags:
            img_file = path.join(outdir, "probabilities.%s.png" % tag)
            csv = path.join(outdir, "evaluate.%s.csv" % tag)
            if not path.isfile(csv):
                continue
            evaluates = pd.read_csv(csv)
            if len(evaluates) == 0:
                continue

            fig, ax = plt.subplots(2, len(events), sharex=True)
            plt.subplots_adjust(wspace=0.5)
            ax = ax.ravel()
            for c, event in enumerate(events):
                positives = evaluates.loc[evaluates[event] == 1,
                                          "%s_prob" % event].values
                negatives = evaluates.loc[evaluates[event] == 0,
                                          "%s_prob" % event].values
                ax[c].hist(positives, bins=bins, color=cmap(0))
                ax[c+len(events)].hist(negatives, bins=bins, color=cmap(1))
                # set x, y labels
                ax[c].set_title(event)
                if c == 0:
                    ax[c].set_ylabel("positive")
                    ax[c+len(events)].set_ylabel("negative")

            a = fig.add_subplot(111, frameon=False)
            a.tick_params(labeltop=False, labelbottom=False,
                          labelleft=False, labelright=False,
                          top=False, bottom=False,
                          left=False, right=False)
            # a.yaxis.set_label_coords(-0.1, 0.5)
            a.xaxis.set_label_coords(0.5, -0.07)
            # a.set_ylabel("positive / negative")
            a.set_xlabel("probability")
            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.10, top=0.95)
            plt.savefig(img_file)
            plt.close()
