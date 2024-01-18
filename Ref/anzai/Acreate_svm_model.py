# -*- coding: utf-8 -*-
import os
from os import path
from argparse import ArgumentParser

import glob
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.externals import joblib

import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

import pandas as pd

def read_csv(accel_file: str, gyro_file: str, label_file: str):
    accels = pd.read_csv(accel_file, parse_dates = [0])
    gyros = pd.read_csv(gyro_file, parse_dates = [0])
    labels = pd.read_csv(label_file, parse_dates = [0])

    xdata = np.hstack([accels[["cepstrum_x00", "cepstrum_x01", "cepstrum_x02", "cepstrum_x03", "cepstrum_x04", "cepstrum_x05", "cepstrum_x06", "cepstrum_x07", "cepstrum_x08", "cepstrum_x09", "cepstrum_x10", "cepstrum_x11", "cepstrum_x12", "cepstrum_x13", "cepstrum_x14", "cepstrum_x15", "cepstrum_x16", "cepstrum_x17", "cepstrum_x18", "cepstrum_x19", "cepstrum_x20", "cepstrum_x21", "cepstrum_x22", "cepstrum_x23", "cepstrum_x24", "cepstrum_x25", "cepstrum_x26", "cepstrum_x27", "cepstrum_x28", "cepstrum_x29"
                               , "cepstrum_y00", "cepstrum_y01", "cepstrum_y02", "cepstrum_y03", "cepstrum_y04", "cepstrum_y05", "cepstrum_y06", "cepstrum_y07", "cepstrum_y08", "cepstrum_y09", "cepstrum_y10", "cepstrum_y11", "cepstrum_y12", "cepstrum_y13", "cepstrum_y14", "cepstrum_y15", "cepstrum_y16", "cepstrum_y17", "cepstrum_y18", "cepstrum_y19", "cepstrum_y20", "cepstrum_y21", "cepstrum_y22", "cepstrum_y23", "cepstrum_y24", "cepstrum_y25", "cepstrum_y26", "cepstrum_y27", "cepstrum_y28", "cepstrum_y29"
                               , "cepstrum_z00", "cepstrum_z01", "cepstrum_z02", "cepstrum_z03", "cepstrum_z04", "cepstrum_z05", "cepstrum_z06", "cepstrum_z07", "cepstrum_z08", "cepstrum_z09", "cepstrum_z10", "cepstrum_z11", "cepstrum_z12", "cepstrum_z13", "cepstrum_z14", "cepstrum_z15", "cepstrum_z16", "cepstrum_z17", "cepstrum_z18", "cepstrum_z19", "cepstrum_z20", "cepstrum_z21", "cepstrum_z22", "cepstrum_z23", "cepstrum_z24", "cepstrum_z25", "cepstrum_z26", "cepstrum_z27", "cepstrum_z28", "cepstrum_z29"]].values
                    , gyros[["cepstrum_x00", "cepstrum_x01", "cepstrum_x02", "cepstrum_x03", "cepstrum_x04", "cepstrum_x05", "cepstrum_x06", "cepstrum_x07", "cepstrum_x08", "cepstrum_x09", "cepstrum_x10", "cepstrum_x11", "cepstrum_x12", "cepstrum_x13", "cepstrum_x14", "cepstrum_x15", "cepstrum_x16", "cepstrum_x17", "cepstrum_x18", "cepstrum_x19", "cepstrum_x20", "cepstrum_x21", "cepstrum_x22", "cepstrum_x23", "cepstrum_x24", "cepstrum_x25", "cepstrum_x26", "cepstrum_x27", "cepstrum_x28", "cepstrum_x29"
                             , "cepstrum_y00", "cepstrum_y01", "cepstrum_y02", "cepstrum_y03", "cepstrum_y04", "cepstrum_y05", "cepstrum_y06", "cepstrum_y07", "cepstrum_y08", "cepstrum_y09", "cepstrum_y10", "cepstrum_y11", "cepstrum_y12", "cepstrum_y13", "cepstrum_y14", "cepstrum_y15", "cepstrum_y16", "cepstrum_y17", "cepstrum_y18", "cepstrum_y19", "cepstrum_y20", "cepstrum_y21", "cepstrum_y22", "cepstrum_y23", "cepstrum_y24", "cepstrum_y25", "cepstrum_y26", "cepstrum_y27", "cepstrum_y28", "cepstrum_y29"
                             , "cepstrum_z00", "cepstrum_z01", "cepstrum_z02", "cepstrum_z03", "cepstrum_z04", "cepstrum_z05", "cepstrum_z06", "cepstrum_z07", "cepstrum_z08", "cepstrum_z09", "cepstrum_z10", "cepstrum_z11", "cepstrum_z12", "cepstrum_z13", "cepstrum_z14", "cepstrum_z15", "cepstrum_z16", "cepstrum_z17", "cepstrum_z18", "cepstrum_z19", "cepstrum_z20", "cepstrum_z21", "cepstrum_z22", "cepstrum_z23", "cepstrum_z24", "cepstrum_z25", "cepstrum_z26", "cepstrum_z27", "cepstrum_z28", "cepstrum_z29"]].values])
    labels.reset_index(drop = True, inplace = True)

    assert xdata.shape[0] == len(labels)

    return xdata, labels

def read_datas(indir: str, car_ids: list, without_no_label: bool=False):
    
    # file_ptn = path.join(dir, "*/*.label.csv")
    #if car_ids is None or len(car_ids) == 0: file_ptns = [path.join(indir, "*.label.csv")]#*/*.label.csv変更
    #elif "/" in car_ids[0]: file_ptns = [s for s in car_ids if '/' in s]#変更
    #else: file_ptns = [path.join(indir, "%s/*.label.csv" % i) for i in car_ids]
    file_ptns = car_ids 
    print(file_ptns)
    xdatas = None
    labels = None

    for file_ptn in file_ptns:
        for _label in glob.glob(file_ptn, recursive = True):
            _accel = _label.replace(".label.csv", ".accel.csv")
            _gyro = _label.replace(".label.csv", ".gyro.csv")
            _xdatas, _labels = read_csv(_accel, _gyro, _label)

            if xdatas is None: xdatas = _xdatas
            else: xdatas = np.vstack([xdatas, _xdatas])
            if labels is None: labels = _labels
            else: labels = pd.concat([labels, _labels], ignore_index = True)
    xlabel = labels[["NO_LABEL", "ROLL", "RUN", "DOOR"]].values
    xlabel = np.argmax(xlabel, axis = -1)
    
    if without_no_label:
        _with_label_idx = np.where(xlabel != 0)
        xlabel = xlabel[_with_label_idx]
        xdatas = xdatas[_with_label_idx]
        del _with_label_idx

    return xdatas, xlabel

def ASVM(indir: str, outmodel: str,car_ids: list, kerrnel: str):
    for combi in combi_list:
        target = [i for i in directory_list if not i in combi]
        xdata, xlabel = Acreate_svm_model.read_datas(indir, combi, without_no_label = without_no_label)
        print("DONE load train data:", xdata.shape, xlabel.shape)
        model = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-3)
        model.fit(xdata, xlabel)
        print("DONE fit")
        outmodel = "./SVMfile/20180814/model/%s.svm"%target[0]
        modeldir = path.dirname(outmodel)
        if not modeldir is None and len(modeldir) > 0 and not path.isdir(modeldir): os.makedirs(modeldir)
        joblib.dump(model, outmodel)
        print(model.score(xdata, xlabel))


