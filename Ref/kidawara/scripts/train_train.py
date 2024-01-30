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

__CAR_IDS__ = [1, 5, 10, 11, 13]
logformat = "[%(asctime)s] <%(levelname)s> %(message)s"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--initial_model", type=str, default=None)
    parser.add_argument("--target_ratio", type=float, default=0)
    parser.add_argument("--trainnc", type=str,
                        default="datas/train.shiei.l8.i100.o000.s100.nc")
    parser.add_argument("--valnc", type=str,
                        default="datas/test.shiei.l8.i100.o000.s100.nc")
    parser.add_argument("--testnc", type=str,
                        default="datas/test.shiei.l8.i100.o000.s100.nc")
    parser.add_argument("--outdir", type = str, default = None)
    #parser.add_argument("--train_car_ids", default=[1,5,10,11],
    #                    type=int, nargs="+", choices=__CAR_IDS__)
    #parser.add_argument("--val_car_ids", default=[1,5,10,11],
    #                    type=int, nargs="+", choices=__CAR_IDS__)
    #parser.add_argument("--test_car_ids", default=[13],
    #                    type=int, nargs="+", choices=__CAR_IDS__)
    parser.add_argument("--initial_epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sample_size", type=int, default=1024) #追加学習のサイズ
    parser.add_argument("--add_size", type=int, default=512)
    parser.add_argument("--data_size", type=int, default=1024) #先行学習のサイズ
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
    parser.add_argument("--auc_lambda", type=float, default=1.0)
    parser.add_argument("--bce_lambda", type=float, default=1.0)
    parser.add_argument("--update_unsertainty_interval", type=int, default=1000)
    parser.add_argument("--unsertainty_event", type=str, default="DOOR",
                        choices=["ROLL", "RUN", "DOOR"])
    parser.add_argument("--loglevel", type=str, default="INFO",
                        choices=loglevels)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log", type=str, default="train_al.log")
    parser.add_argument("--eval_size", type=int, default = 0)
    args = parser.parse_args()
    config_logger(logformat, args.loglevel, args.logfile)
    logger = getLogger(__name__)
    logger.info(args)


    for i in range(1):
        params = vars(args).copy()
        test_car_ids = [__CAR_IDS__[i]]
        train_car_ids= __CAR_IDS__[:i]+__CAR_IDS__[i+1:]
        params["train_car_ids"]=train_car_ids
        params["val_car_ids"]=train_car_ids
        params["test_car_ids"] =test_car_ids

        params["outdir_initial_train"]=("cat_results\\results_car_id_%s_initial" %test_car_ids)
        params["outdir_additional_train"]=("cat_results\\result_car_id_%ss_additional" %test_car_ids)


        # del positional args
        del params["trainnc"], params["valnc"], params["testnc"], params["outdir"], params["eval_size"]

        criterion_conf = OrderedDict([("AUC", args.auc_lambda),
                                      ("BCE", args.bce_lambda)])
        del params["auc_lambda"], params["bce_lambda"]
        # logger.debug(params)
        args.outdir = params["outdir_initial_train"]

        trainer = AUCTrainer(cuda=args.cuda)
        trainer.train(args.trainnc, args.valnc, args.testnc, args.outdir,
                      criterion_config=criterion_conf, **params)
        trainer.evaluate(args.trainnc, args.trainnc, args.testnc, args.outdir, args.eval_size,
                         **params)
        trainer.plot(args.outdir, **params)

        #additional_learning
        params["initial_model"] = path.join(params["outdir_initial_train"], "best_model.pth")
        args.outdir = params["outdir_initial_train"]
        params["sample_size"] = params["add_size"]
        params["initial_epoch"] =151
        params["epoch"] =1
        trainer = UnsertaintyTrainer(cuda=args.cuda)
        trainer.train(args.trainnc, args.valnc, args.testnc, args.outdir,
                      criterion_config=criterion_conf, **params)
        trainer.evaluate(args.trainnc, args.trainnc, args.testnc, args.outdir, args.eval_size,
                         **params)
        trainer.plot(args.outdir, **params)
