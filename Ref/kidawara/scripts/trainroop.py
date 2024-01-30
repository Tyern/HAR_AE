#! -*- coding: utf-8
import os
import os.path as path
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch

from datas import *
from losses import *
from models import *
from trainer import *
from logs import loglevels, config_logger
from logging import getLogger
from statistics import mean

__CAR_IDS__ = [1, 5, 10, 11, 13]
logformat = "[%(asctime)s] <%(levelname)s> %(message)s"

# desktop/能動学習/pic/20200907


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--initial_model", type=str, default=None)
    parser.add_argument("--trainnc", type=str,
                        default="datas/train.shiei.l8.i100.o000.s100.nc")
    parser.add_argument("--valnc", type=str,
                        default="datas/train.shiei.l8.i100.o000.s100.nc")
    parser.add_argument("--testnc", type=str,
                        default="datas/test.shiei.l8.i100.o000.s100.nc")
    parser.add_argument("--outdir", type=str,
                        default="results")
    parser.add_argument("--train_car_ids", default=[1, 5, 10],
                        type=int, nargs="+", choices=__CAR_IDS__)
    parser.add_argument("--val_car_ids", default=[11],
                        type=int, nargs="+", choices=__CAR_IDS__)
    parser.add_argument("--test_car_ids", default=[13],
                        type=int, nargs="+", choices=__CAR_IDS__)
    parser.add_argument("--initial_epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--events", type=str, nargs="+",
                        default=["ROLL", "RUN", "DOOR"],
                        choices=["ROLL", "RUN", "DOOR"])
    parser.add_argument("--common_channels",
                        type=int, nargs="+", default=[32, 64])
    parser.add_argument("--common_kernel_size",
                        type=int, nargs=2, default=(8, 1))
    parser.add_argument("--common_strides",
                        type=int, nargs=2, default=(3, 1))
    parser.add_argument("--event_channels",
                        type=int, nargs="+", default=[64])
    parser.add_argument("--event_kernel_size",
                        type=int, nargs=2, default=(1, 8))
    parser.add_argument("--event_strides",
                        type=int, nargs=2, default=(1, 1))
    parser.add_argument("--event_features",
                        type=int, nargs="+", default=[128, 32])
    parser.add_argument("--auc_lambda", type=float, default=1.0) #0:使用しない
    parser.add_argument("--bce_lambda", type=float, default=1.0) #同上
    parser.add_argument("--loglevel", type=str, default="INFO",
                        choices=loglevels)
    parser.add_argument("--logfile", type=str, default=None) #default="PATH"でlogデータを保存
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    config_logger(logformat, args.loglevel, args.logfile)
    logger = getLogger(__name__)
    logger.info(args)

    params = vars(args).copy()
    # del positional args
    del params["trainnc"], params["valnc"], params["testnc"], params["outdir"]
    criterion_conf = OrderedDict([("AUC", args.auc_lambda),
                                  ("BCE", args.bce_lambda)])
    del params["auc_lambda"], params["bce_lambda"]
    # logger.debug(params)
    """
    train_dataset = SensorDataset(args.trainnc, args.train_car_ids)
    add_dataset = SensorDataset(args.testnc, args.test_car_ids)
    print(len(add_dataset))
    cat_dataset = CatDataset(train_dataset, add_dataset)

    trained_data = pd.read_csv("results\\train.csv")
    trained_idx =trained_data.loc[:,"original"].values
    train_dataset.events[trained_idx].sum()
    #print(add_dataset.car_ids.shape)
    print(cat_dataset[50000])
    print(cat_dataset.car_ids.shape)

    Door_True = dataset.events[:,2]>dataset.thresholds["DOOR"]
    #Door_False = 1- Door_True
    Door_Tind = np.nonzero(Door_True.numpy())
    print(len(Door_Tind[0]))
    """
    trainer = RoopTrainer(cuda=args.cuda)
    #train_sizes = [128, 256]
    train_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 38089]
    trainrange=10
    #for event in args.events:
    #    "%s_Scores" %event : OrderedDict([("AUC", []), ("mic_ACC", []), ("mac_ACC", [])])
    ROLL_score = OrderedDict([("AUC", {}), ("mic_ACC", {}), ("mac_ACC", {})])
    RUN_score = OrderedDict([("AUC", {}), ("mic_ACC", {}), ("mac_ACC", {})])
    DOOR_score = OrderedDict([("AUC", {}), ("mic_ACC", {}), ("mac_ACC", {})])
    TrainEvents={}
    for train_size in train_sizes:
        print("trianing size = %d" %train_size)
        #for event in args.events:
        #    for score in ["AUC", "mic_ACC", "mac_ACC"]:
        #        "%s_Scores[%s][train_size]" %(event, score) : []
        for score in ["AUC", "mic_ACC", "mac_ACC"]:
            ROLL_score[score][train_size] = []
            RUN_score[score][train_size] = []
            DOOR_score[score][train_size] = []
        Train_events=[]
        for i in range(trainrange):
            print("train %d" %(i+1))
            trainer.train(args.trainnc, args.valnc, args.testnc, args.outdir, train_size=train_size,
                          criterion_config=criterion_conf, **params)
            print(sum(trainer.train_events))
            Train_events.append(sum(trainer.train_events))
            del trainer.train_events
            AUC, mic_ACC, mac_ACC = trainer.evaluate(args.trainnc, args.trainnc, args.testnc, args.outdir,
                             **params)
            for score_name, score in zip(["AUC", "mic_ACC", "mac_ACC"], [AUC, mic_ACC, mac_ACC]):
                ROLL_score[score_name][train_size].append(score["ROLL"])
                RUN_score[score_name][train_size].append(score["RUN"])
                DOOR_score[score_name][train_size].append(score["DOOR"])
            #ROLL_score[train_size].append(AUC["ROLL"])
            #RUN_score[train_size].append(AUC["RUN"])
            #DOOR_score[train_size].append(AUC["DOOR"])
        TrainEvents[train_size]=Train_events
        del Train_events
        print("training size = %d: mean AUC [ROLL] = %.3f" %(train_size, mean(ROLL_score["AUC"][train_size])))
        print("training size = %d: mean AUC [RUN] = %.3f" %(train_size, mean(RUN_score["AUC"][train_size])))
        print("training size = %d: mean AUC [DOOR] = %.3f" %(train_size, mean(DOOR_score["AUC"][train_size])))
    Scores = {"ROLL":ROLL_score, "RUN":RUN_score, "DOOR":DOOR_score}
    print(Scores)
    print(TrainEvents)
    trainer.plot(args.outdir, train_sizes, Scores, **params)
