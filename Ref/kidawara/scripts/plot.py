import os
import os.path as path
from argparse import ArgumentParser
from collections import OrderedDict

import torch

from datas import *
from losses import *
from models import *
from trainer import *
from logs import loglevels, config_logger
from logging import getLogger

__CAR_IDS__ = [1, 5, 10, 11, 13]
logformat = "[%(asctime)s] <%(levelname)s> %(message)s"

parser = ArgumentParser()
parser.add_argument("initial_model", type=str)
parser.add_argument("--trainnc", type=str,
                    default="datas/train.shiei.l8.i100.o000.s100.nc")
parser.add_argument("--valnc", type=str,
                    default="datas/test.shiei.l8.i100.o000.s100.nc")
parser.add_argument("--testnc", type=str,
                    default="datas/test.shiei.l8.i100.o000.s100.nc")
parser.add_argument("--outdir", type=str,
                    default="results")
parser.add_argument("--train_car_ids", default=[11],
                    type=int, nargs="+", choices=__CAR_IDS__)
parser.add_argument("--val_car_ids", default=[11],
                    type=int, nargs="+", choices=__CAR_IDS__)
parser.add_argument("--test_car_ids", default=[13],
                    type=int, nargs="+", choices=__CAR_IDS__)
parser.add_argument("--initial_epoch", type=int, default=101)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--sample_size", type=int, default=1024) #追加学習のサイズ
parser.add_argument("--data_size", type=int, default=1024)
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
parser.add_argument("--update_unsertainty_interval", type=int, default=100)
parser.add_argument("--unsertainty_event", type=str, default="ROLL",
                    choices=["ROLL", "RUN", "DOOR"])
parser.add_argument("--loglevel", type=str, default="INFO",
                    choices=loglevels)
parser.add_argument("--logfile", type=str, default=None)
parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()

params = vars(args).copy()
del params["trainnc"], params["valnc"], params["testnc"], params["outdir"]

trainer = AUCTrainer(cuda=args.cuda)
#trainer.evaluate(args.trainnc, args.trainnc, args.testnc, args.outdir,
#                 **params)
trainer.plot(args.outdir, **params)
