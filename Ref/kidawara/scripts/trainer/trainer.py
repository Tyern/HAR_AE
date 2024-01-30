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
import random

CarIds = typing.List[int]
CriterionFactor = typing.Tuple[torch.nn.Module, float]
Criterions = typing.OrderedDict[str, CriterionFactor]

__all__ = ["AUCTrainer"]


class AUCTrainer(Trainer):
    def __init__(self,
                 cuda: bool = False):
        super(AUCTrainer, self).__init__(cuda=cuda)


    """def DataLoader(self, ncfile: str,
                   car_ids: CarIds = None,
                   sample_size: int = 1024,
                   batch_size: int = 1024,
                   data_size: int = 0,
                   shuffle: bool = False,
                   **kwargs):
        assert path.isfile(ncfile)
        assert batch_size > 0
        assert car_ids is None or len(car_ids) > 0
        sampler = RandomSampler(dataset= ncfile,
                                sample_size = sample_size)

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           sampler = sampler
                                           shuffle=shuffle)"""

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
        """ 最終層が線形となるモデルを選択
        """
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
        #if isinstance(initial_model, torch.nn.Module):
        try:
            model.load_state_dict(initial_model)
        except:
            pass
        else:
            print("model loaded")
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
                eps = 1e-7
                #cutoffs = np.sqrt((1-tpr)**2 + fpr**2)
                #threshold = thresholds[np.argmin(cutoffs)]
                cutoffs = 1/(0.5 * (1/(tpr+eps) + 1/(1-fpr+eps)))
                threshold = thresholds[np.argmax(cutoffs)]
                #cutoffs = 0.5*(tpr + (1-fpr))
                #threshold = thresholds[np.argmax(cutoffs)]
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
              sample_size :int = 1024,
              train_car_ids: CarIds = [1, 5, 10],
              val_car_ids: CarIds = [11],
              test_car_ids: CarIds = [13],
              epochs: int = 1000,
              initial_epoch: int = 1,
              initial_model: str = None,
              seed: int = 11,
              criterion_config: OrderedDict = OrderedDict([("AUC", 1.0),
                                                           ("BCE", 1.0)]),
              **kwargs):
        self.sampler =RandSampler(train_file)
        #self.sampler =BiasSampler(train_file) #DOORの数を操作する際に使用
        if seed is not None and seed > 0:
            torch.manual_seed(seed)
            random.seed(seed) #random.sampleのシード値固定
            pass
        train_loader = self.DataLoader(train_file,
                                       car_ids=train_car_ids,
                                       sampler=self.sampler,
                                       sampler_size=sample_size,
                                       shuffle=False, **kwargs)
        os.makedirs(outdir, exist_ok=True)
        with open(path.join(outdir, "train.csv"), "wt") as f:
            f.write("original")
            f.write("\n")
            for n in train_loader.sampler:
                f.write(str(n))
                f.write("\n")
        self.train_events = []
        self.double_events = OrderedDict([("ROLL.RUN", 0), ("ROLL.DOOR", 0), ("RUN.DOOR", 0), ("ALL", 0)])
        for _, events in train_loader:
            #events = events.to(self.device)
            self.train_events.append(events.detach().cpu().sum(dim=0))
            doubeve = events.numpy().copy()*[3,5,7]
            self.double_events["ROLL.RUN"] = ((doubeve.sum(axis=1)==8)*1).sum()
            self.double_events["ROLL.DOOR"] = ((doubeve.sum(axis=1)==10)*1).sum()
            self.double_events["RUN.DOOR"] = ((doubeve.sum(axis=1)==12)*1).sum()
            self.double_events["ALL"] = ((doubeve.sum(axis=1)==15)*1).sum()

        print(self.train_events)
        print(self.double_events)


        #print(self.trained_idx)
        val_loader = self.DataLoader(val_file,
                                     car_ids=val_car_ids,
                                     shuffle=True, **kwargs)
        test_loader = self.DataLoader(test_file,
                                      car_ids=test_car_ids,
                                      shuffle=False, **kwargs)

        os.makedirs(outdir, exist_ok=True)
        best_model_file = path.join(outdir, "best_model.pth")
        last_model_file = path.join(outdir, "last_model.pth")
        train_log_file = path.join(outdir, "train.log")

        model = self.Model(initial_model=initial_model, **kwargs)
        model.to(self.device)

        """i=0
        for p in (model.parameters("common_layers")):
            i+=1
            if i==5:
                break
            else:
                p.requires_grad = False"""

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

        torch.save(model.state_dict(), last_model_file)
        if not path.isfile(best_model_file):
            torch.save(model.state_dict(), best_model_file)



    def evaluate(self,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 outdir: str,
                 sampler = None,
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
                                      car_ids=train_car_ids, shuffle=True, **kwargs)),
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
                       "accuracy", "precision", "recall", "auc", "threshold", "nearest100", "nearest200","nearest500", "nearest1000" ]
            f.write(",".join(headers))
            f.write("\n")
            for tag, eval_file in eval_files.items():
                eval = pd.read_csv(eval_file)
                for event in model.events():
                    t = eval.loc[:, event].values
                    outs = eval.loc[:, "%s_prob" % event].values

                    fpr, tpr, thresholds = roc_curve(t, outs, pos_label=1)
                    auc_score = auc(fpr, tpr)
                    """plt.fill_between(fpr, 0, tpr)
                    plt.plot(fpr, tpr)
                    plt.xlim(0.,1.)
                    plt.ylim(0.,1.)
                    plt.show()"""
                    eps = 1e-7

                    """
                    #accuracyによる閾値を求めるエリア
                    #ヒストグラムに赤線を入れるための下準備
                    HITS=[]
                    for i, th in enumerate(thresholds):
                        hits = sum(1-(t- (outs>=th).astype(float))**2)
                        HITS.append(int(hits))
                    acc_th=thresholds[np.argmax(HITS)]
                    """

                    #acc_th=None
                    # cutoffs
                    #cutoffs = np.sqrt((1-tpr)**2 + fpr**2)
                    #threshold = thresholds[np.argmin(cutoffs)]
                    cutoffs = 1/(0.5*(1/(tpr+eps) + 1/(1-fpr+eps)))
                    threshold = thresholds[np.argmax(cutoffs)]
                    #cutoffs = 0.5*(tpr + (1-fpr))
                    #threshold = thresholds[np.argmax(cutoffs)]


                    preds = (outs >= threshold).astype(float)
                    hits = 1-(t-preds)**2
                    miss = 1-hits


                    ###閾値近傍を見るエリア
                    uncertainty = abs(outs - threshold).astype(float)
                    positive_uncertainty = uncertainty + (outs<threshold)*1e2
                    uncertain_idx = uncertainty.argsort()
                    certain_idx = uncertainty.argsort()[::-1]
                    positive_idx = np.random.permutation(preds.nonzero()).ravel()

                    positive_uncertain_idx = positive_uncertainty.argsort()
                    #print(sum(t[positive_uncertain_idx[:100]]))
                    #print(outs[uncertain_idx])
                    #print(threshold, uncertainty[uncertain_idx[0]], outs[uncertain_idx[0]])
                    nearest_range = [100, 200, 500, 1000]
                    uncertain_True_in_nearest = OrderedDict([(n_range, 0) for n_range in nearest_range])
                    for n in nearest_range:
                        uncertain_True_in_nearest[n] = sum(t[positive_idx[:n]])
                    ###

                    tp, tn = int(np.sum(t * hits)), int(np.sum((1-t)*hits))
                    fp, fn = int(np.sum((1-t) * miss)), int(np.sum(t * miss))
                    length = len(t)
                    t = np.sum(t)
                    preds = np.sum(preds)

                    line = [tag, event, str(t), str(length),
                            str(tp), str(tn), str(fp), str(fn),
                            "%0.6f" % (np.sum(hits)/length),
                            "%0.6f" % (tp/preds),
                            "%0.6f" % (tp/t),
                            "%0.6f" % auc_score,
                            "%0.6f" % threshold] + [str(uncertain_True_in_nearest[NumofTrue]) for NumofTrue in nearest_range]#+ [str(acc_th)]
                    f.write(",".join(line))
                    f.write("\n")
                    print("%s: AUC = %.3f, microACC = %.3f, macroACC = %.3f" %(event, auc_score, (tp+tn)/(tp+tn+fp+fn), (tp/(tp+fn)+tn/(tn+fp))*0.5))
                    if tag=="test":
                        print("%s: [%.3f, %.3f, %.3f]" %(event, auc_score, (tp+tn)/(tp+tn+fp+fn), (tp/(tp+fn)+tn/(tn+fp))*0.5))


    def plot(self, outdir: str, **kwargs):
        train_log_file = path.join(outdir, "train.log")
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
        thresholds = OrderedDict()
        acc_th = OrderedDict()
        for i, row in scores.iterrows():
            thresholds[(row.tag, row.event)] = row.threshold
            #acc_th[(row.tag, row.event)] = row.acc_threshold
        bins = 100
        cmap = plt.get_cmap("tab10")
        PosOverThreshSize = OrderedDict([("ROLL", 0), ("RUN", 0), ("DOOR", 0)])
        NegOverThreshSize = OrderedDict([("ROLL", 0), ("RUN", 0), ("DOOR", 0)])
        for tag in tags:
            img_file = path.join(outdir, "probabilities.%s.png" % tag)
            csv = path.join(outdir, "evaluate.%s.csv" % tag)
            if not path.isfile(csv):
                continue
            evaluates = pd.read_csv(csv)
            if len(evaluates) == 0:
                continue

            fig, ax = plt.subplots(2, len(events))
            plt.subplots_adjust(wspace=0.5)
            ax = ax.ravel()
            for c, event in enumerate(events):
                t = thresholds[(tag, event)]
                at = acc_th[(tag, event)]
                positives = evaluates.loc[evaluates[event] == 1,
                                          "%s_prob" % event].values

                negatives = evaluates.loc[evaluates[event] == 0,
                                          "%s_prob" % event].values
                outs = evaluates.loc[:,
                                          "%s_prob" % event].values
                evaluates["%s_unsertainty"%event] = abs(evaluates["%s_prob"%event] - t)
                eval_sorted = evaluates.sort_values(by="%s_unsertainty"%event, ascending=True)
                #print(sum(eval_sorted[event][:100]))
                #print(sum(evaluates[event][:500]))
                #ax[c].hist(negatives,bins=bins, color=cmap(1), align="left")

                x, y, pos1 = ax[c].hist(positives,
                                        bins=bins, color=cmap(0), align="left")

                ax[c].set_ylim([0., x.max()])
                if tag == "test":

                    PosOverThreshSize[event] += (x * (y[1:]>=t)).sum()



                ax[c].vlines(x=t, ymin=0, ymax=x.max(), color = "black")
                ax[c].vlines(x=at, ymin=0, ymax=x.max(), color = "red")
                x, y, pos2 = ax[c+len(events)].hist(negatives,
                                                    bins=bins, color=cmap(1), align="left")
                ax[c+len(events)].set_ylim([0., x.max()])
                #ax[c+len(events)].hist(positives, bins=bins, color=cmap(0), align="left")

                if tag == "test":
                    NegOverThreshSize[event] += (x * (y[1:]>=t)).sum()


                # hist のプロット情報から各イベントごとのヒストグラムの左座標、右座標を算出する
                w1 = pos1[1].xy[0] - pos1[0].xy[0]  # バーの幅を計算
                w2 = pos2[1].xy[0] - pos2[0].xy[0]  # バーの幅を計算
                # 各ヒストグラムのバーの左下座標が記録されている
                left = np.min([pos1[0].xy[0] - w1*0.5,
                               pos2[0].xy[0] - w2*0.5])  # バーの幅半分の余白作る
                right = np.max([pos1[-1].xy[0] + w1*1.5,
                                pos2[-1].xy[0] + w2*1.5])  # バーの幅半分の余白作る
                ax[c].set_xlim(left, right)
                ax[c+len(events)].set_xlim(left, right)

                ax[c+len(events)].vlines(x=t, ymin=0, ymax=x.max(), color = "Black")
                ax[c+len(events)].vlines(x=at, ymin=0, ymax=x.max(), color = "red")

                # set x, y labels
                ax[c].set_title(event)
                if c == 0:
                    ax[c].set_ylabel("positive (+ negative)")
                    ax[c+len(events)].set_ylabel("negative (+ positive)")

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


        #print("Positive: %s, \n Negative: %s" %(PosOverThreshSize, NegOverThreshSize))
        ##### 閾値近傍2000件をプロットする #####
        nplots = 2000
        for tag in tags:
            img_file = path.join(outdir, "unsertainty.%s.png" % tag)
            csv = path.join(outdir, "evaluate.%s.csv" % tag)
            if not path.isfile(csv):
                continue
            evaluates = pd.read_csv(csv)
            if len(evaluates) == 0:
                continue

            fig, ax = plt.subplots(len(events), len(events))
            plt.subplots_adjust(wspace=0.5)
            ax = ax.ravel()
            for r, target_event in enumerate(events):
                # set unsertainty
                samples = evaluates.copy()
                threshold = thresholds[(tag, target_event)]
                samples["unsertainty"] = np.abs(
                    samples["%s_prob" % target_event].values - threshold)

                # select samples
                samples.sort_values(by="unsertainty",
                                    ascending=True, inplace=True)
                samples.reset_index(inplace=True, drop=True)
                samples = samples.iloc[:nplots]

                for c, event in enumerate(events):
                    # plot samples
                    figidx = r*len(events) + c
                    positives = samples.loc[samples[event] == 1,
                                            "%s_prob" % event].values
                    negatives = samples.loc[samples[event] == 0,
                                            "%s_prob" % event].values
                    ax[figidx].scatter(
                        positives, [0.01]*len(positives), s=0.1, label="positive")
                    ax[figidx].scatter(
                        negatives, [-0.01]*len(negatives), s=0.1, label="negative")
                    ax[figidx].vlines(x=thresholds[(tag, event)],
                                      ymin=-0.1, ymax=0.1)

                    if r == 0:
                        ax[figidx].set_title(event)
                    if c == 0:
                        ax[figidx].set_ylabel(target_event)
                    #print("tag: %s, target_event: %s, selected samples: %s, positive: %d, negative: %d" %(tag, target_event,  event, len(positives), len(negatives)))

            a = fig.add_subplot(111, frameon=False)
            a.tick_params(labeltop=False, labelbottom=False,
                          labelleft=False, labelright=False,
                          top=False, bottom=False,
                          left=False, right=False)
            a.yaxis.set_label_coords(-0.1, 0.5)
            a.xaxis.set_label_coords(0.5, -0.07)
            a.set_ylabel("target event")
            a.set_xlabel("selected samples")
            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.10, top=0.95)
            plt.savefig(img_file)
            plt.close()
