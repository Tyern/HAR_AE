# -*- coding: utf-8 -*-
import os
from os import path
from argparse import ArgumentParser

import glob
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils

import RNN

def read_csv(accel_file: str, gyro_file: str, label_file: str, timestep: int, timeshift: int):
    accels = pd.read_csv(accel_file, parse_dates = [0])
    gyros = pd.read_csv(gyro_file, parse_dates = [0])
    labels = pd.read_csv(label_file, parse_dates = [0])

    xdata = np.hstack([accels[["cepstrum_x00", "cepstrum_x01", "cepstrum_x02", "cepstrum_x03", "cepstrum_x04", "cepstrum_x05", "cepstrum_x06", "cepstrum_x07", "cepstrum_x08", "cepstrum_x09", "cepstrum_x10", "cepstrum_x11", "cepstrum_x12", "cepstrum_x13", "cepstrum_x14", "cepstrum_x15", "cepstrum_x16", "cepstrum_x17", "cepstrum_x18", "cepstrum_x19", "cepstrum_x20", "cepstrum_x21", "cepstrum_x22", "cepstrum_x23", "cepstrum_x24", "cepstrum_x25", "cepstrum_x26", "cepstrum_x27", "cepstrum_x28", "cepstrum_x29"
                               , "cepstrum_y00", "cepstrum_y01", "cepstrum_y02", "cepstrum_y03", "cepstrum_y04", "cepstrum_y05", "cepstrum_y06", "cepstrum_y07", "cepstrum_y08", "cepstrum_y09", "cepstrum_y10", "cepstrum_y11", "cepstrum_y12", "cepstrum_y13", "cepstrum_y14", "cepstrum_y15", "cepstrum_y16", "cepstrum_y17", "cepstrum_y18", "cepstrum_y19", "cepstrum_y20", "cepstrum_y21", "cepstrum_y22", "cepstrum_y23", "cepstrum_y24", "cepstrum_y25", "cepstrum_y26", "cepstrum_y27", "cepstrum_y28", "cepstrum_y29"
                               , "cepstrum_z00", "cepstrum_z01", "cepstrum_z02", "cepstrum_z03", "cepstrum_z04", "cepstrum_z05", "cepstrum_z06", "cepstrum_z07", "cepstrum_z08", "cepstrum_z09", "cepstrum_z10", "cepstrum_z11", "cepstrum_z12", "cepstrum_z13", "cepstrum_z14", "cepstrum_z15", "cepstrum_z16", "cepstrum_z17", "cepstrum_z18", "cepstrum_z19", "cepstrum_z20", "cepstrum_z21", "cepstrum_z22", "cepstrum_z23", "cepstrum_z24", "cepstrum_z25", "cepstrum_z26", "cepstrum_z27", "cepstrum_z28", "cepstrum_z29"]].values
                    , gyros[["cepstrum_x00", "cepstrum_x01", "cepstrum_x02", "cepstrum_x03", "cepstrum_x04", "cepstrum_x05", "cepstrum_x06", "cepstrum_x07", "cepstrum_x08", "cepstrum_x09", "cepstrum_x10", "cepstrum_x11", "cepstrum_x12", "cepstrum_x13", "cepstrum_x14", "cepstrum_x15", "cepstrum_x16", "cepstrum_x17", "cepstrum_x18", "cepstrum_x19", "cepstrum_x20", "cepstrum_x21", "cepstrum_x22", "cepstrum_x23", "cepstrum_x24", "cepstrum_x25", "cepstrum_x26", "cepstrum_x27", "cepstrum_x28", "cepstrum_x29"
                             , "cepstrum_y00", "cepstrum_y01", "cepstrum_y02", "cepstrum_y03", "cepstrum_y04", "cepstrum_y05", "cepstrum_y06", "cepstrum_y07", "cepstrum_y08", "cepstrum_y09", "cepstrum_y10", "cepstrum_y11", "cepstrum_y12", "cepstrum_y13", "cepstrum_y14", "cepstrum_y15", "cepstrum_y16", "cepstrum_y17", "cepstrum_y18", "cepstrum_y19", "cepstrum_y20", "cepstrum_y21", "cepstrum_y22", "cepstrum_y23", "cepstrum_y24", "cepstrum_y25", "cepstrum_y26", "cepstrum_y27", "cepstrum_y28", "cepstrum_y29"
                             , "cepstrum_z00", "cepstrum_z01", "cepstrum_z02", "cepstrum_z03", "cepstrum_z04", "cepstrum_z05", "cepstrum_z06", "cepstrum_z07", "cepstrum_z08", "cepstrum_z09", "cepstrum_z10", "cepstrum_z11", "cepstrum_z12", "cepstrum_z13", "cepstrum_z14", "cepstrum_z15", "cepstrum_z16", "cepstrum_z17", "cepstrum_z18", "cepstrum_z19", "cepstrum_z20", "cepstrum_z21", "cepstrum_z22", "cepstrum_z23", "cepstrum_z24", "cepstrum_z25", "cepstrum_z26", "cepstrum_z27", "cepstrum_z28", "cepstrum_z29"]].values])
    labels = labels.loc[range(timestep - 1, len(labels), timeshift)]
    labels.reset_index(drop = True, inplace = True)

    # ndim = 2, dtype = float to ndim = 3, dtype = float
    size_f = xdata.dtype.itemsize # dtypeのバイトサイズ
    strided_shape = ((xdata.shape[0] - timestep + timeshift) // timeshift, timestep, xdata.shape[1]) # stride後の次元数を計算
    # stride後のndarrayでxdataを上書き
    xdata = as_strided(xdata, shape = strided_shape, strides = (timeshift * size_f, xdata.strides[0], xdata.strides[1]))

    assert xdata.shape[0] == len(labels)

    return xdata, labels

def read_datas(indir: str, car_ids: list, timestep: int, timeshift: int):

    # file_ptn = path.join(dir, "*/*.label.csv")
    if car_ids is None or len(car_ids) == 0: file_ptns = [path.join(indir, "*.label.csv")]#*/*.label.csv変更
    elif "/" in car_ids[0]: file_ptns = [s for s in car_ids if '/' in s]#変更
    else: file_ptns = [path.join(indir, "%s/*.label.csv" % i) for i in car_ids]
    print(file_ptns)
    xdatas = None
    labels = None

    for file_ptn in file_ptns:
        for _label in glob.glob(file_ptn, recursive = True):
            #if "20170407" in _label or "20170512" in _label: continue # 20170406 もしくは 20170511を訓練データとする
            _accel = _label.replace(".label.csv", ".accel.csv")
            _gyro = _label.replace(".label.csv", ".gyro.csv")
            _xdatas, _labels = read_csv(_accel, _gyro, _label, timestep, timeshift)
            #データの結合
            if xdatas is None: xdatas = _xdatas
            else: xdatas = np.vstack([xdatas, _xdatas])
            if labels is None: labels = _labels
            else: labels = pd.concat([labels, _labels], ignore_index = True)
    xlabel = labels[["NO_LABEL", "ROLL", "RUN", "DOOR"]].values
    xlabel = np.argmax(xlabel, axis = -1)#axis = -1の意味がわからない
    labels["label"] = xlabel
    #if len(labels) > 10000: labels = labels.sample(n = 10000, replace = True, random_state=0)#labelの行が10000を超えたらランダムで抽出(radomstateで固定)
    
    yidxs = labels.index.values
    _labels = labels[["car_id", "label", "timestamp"]]
    _labels["date"] = np.datetime_as_string(labels.timestamp, "D")#timeごとに整列化
    #_labels.to_csv(metapath, sep = "\t", index_label = "row_no")
    xlabel = np_utils.to_categorical(xlabel, num_classes = 4)#ベクトル化
    return xdatas, xlabel, xdatas[yidxs], xlabel[yidxs]

def test2main(indir: str, outmodel: str, outeval:str, outweights: str,outpredict: str, batch_size: int, epochs: int, units: int, timestep: int, timeshift:int, car_ids: list, eval_car_ids: list):
    
    '''
    parser = ArgumentParser()
    parser.add_argument(indir, "-i", required = True, type = str, help = "入力ファイルの配置ディレクトリ")
    parser.add_argument(outmodel, "-o", required = True, type = str, help = "出力モデル")
    parser.add_argument(outeval, "-O", required = True, type = str, help = "出力評価ファイル")
    #parser.add_argument("--logdir", "-l", required = True, type = str, help = "ログ出力ディレクトリ")
    parser.add_argument("-batch_size", default = 100, type = int, help = "バッチサイズ")
    parser.add_argument("-epochs", default = 40, type = int, help = "エポック数")
    parser.add_argument("-units", default = 180, type = int, help = "ユニット数")
    parser.add_argument("-timestep", default = 3, type = int, help = "LSTMの忘却ゲートが有効な期間")
    parser.add_argument("-timeshift", default = 1, type = int, help = "ケプストラムデータのステップ数")
    parser.add_argument("-car_ids", default = None, nargs = "*", type = int, help = "対象とする車両ID。未指定の場合、全車両を対象とする")
    parser.add_argument(eval_car_ids, default = None, nargs = "*", type = int, help = "対象とする車両ID。未指定の場合、全車両を対象とする")
    
    args = parser.parse_args()
    '''
    # if len(args.logdir) > 0 and not path.isdir(args.logdir): os.makedirs(args.logdir)
    # metafile = "metafile.tsv"
    # metapath = path.join(args.logdir, metafile)
    
    #timestep = args.timestep
    #timestep = 3
    #xdata, xlabel, ydata, ylabel = read_datas(args.indir, args.car_ids, args.timestep, args.timeshift)
    xdata, xlabel, ydata, ylabel = read_datas(indir, car_ids, timestep, timeshift)

    #GPUの設定
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    #------------------------------------------
    try:
        #model = RNN.get_model(name = "ntt_sensor_data_from_spectrum", units = args.units, batch_size = args.batch_size, time_step = timestep, feature_dim = xdata.shape[-1], label_size = xlabel.shape[-1])
        model = RNN.get_model(name = "ntt_sensor_data_from_spectrum", units = units, batch_size = batch_size, time_step = timestep, feature_dim = xdata.shape[-1], label_size = xlabel.shape[-1])
        model.summary()

        # class weightを設定する
        _class_weight = {}
        _label_counts = np.sum(xlabel, axis = 0)
        _max_count = np.max(_label_counts)
        for _idx, _count in enumerate(_label_counts): _class_weight[_idx] = _max_count / _count if _count > 0 else 0

        #model.fit(xdata, xlabel, batch_size = args.batch_size, epochs = args.epochs, callbacks = []
        #          , validation_data = [ydata, ylabel], class_weight = _class_weight
        #          , verbose = 2)
        model.fit(xdata, xlabel, batch_size = 100, epochs = epochs, callbacks = []
                  , validation_data = [ydata, ylabel], class_weight = _class_weight
                  , verbose = 2)
        #tensorboard = TensorBoard(log_dir = args.logdir, batch_size = args.batch_size, histogram_freq = 1
        #                          , embeddings_freq = args.epochs - 1, embeddings_layer_names = ["LSTM", "Dense3"]
        #                          , embeddings_metadata = metafile, val_size = len(ylabel), img_path = None, img_size = None)
        #model.fit(xdata, xlabel, batch_size = args.batch_size, epochs = args.epochs, callbacks = [tensorboard]
        #          , validation_data = [ydata, ylabel]
        #          , verbose = 2)

        #outdir = path.dirname(args.outmodel)
        outdir = path.dirname(outmodel)
        if not outdir is None and len(outdir) > 0 and not path.isdir(outdir): os.makedirs(outdir)
        model.save(outmodel)
        model.save_weights(outweights)
        #predicted = model.predict(ydata, batch_size = args.batch_size)
        #predicted = np.argmax(predicted, axis = -1)
        #add_predicted(metapath, predicted)
        
        #評価
        with open(outeval, "wt") as f:#変更
            _headers = ["accel", "gyro", "label","data_counts", "loss", "accuracy", "macro-fmesure"]
            _headers += ["NO_LABEL", "ROLL", "RUN", "DOOR"]
            f.write(",".join(_headers))
            f.write("\n")
            #if args.eval_car_ids is None or len(args.eval_car_ids) == 0: file_ptns = [path.join(args.indir, "*/*.label.csv")]
            if eval_car_ids is None or len(eval_car_ids) == 0: file_ptns = [path.join(indir, "*/*.label.csv")]
            #else: file_ptns = [path.join(args.indir, "%03d/*.label.csv" % i) for i in args.eval_car_ids]
            else: file_ptns = [path.join(indir, "%s/*.label.csv" % i) for i in eval_car_ids]
            for _ptn in file_ptns:
                for _label in glob.glob(_ptn, recursive = True):
                    # if "20170406" in _label or "20170511" in _label: continue # 20170406 もしくは 20170511を訓練データとする
                    _accel = _label.replace("label", "accel")
                    _gyro = _label.replace("label", "gyro")
                    #read_csv(accel_file: str, gyro_file: str, label_file: str, timestep: int, timeshift: int)
                    _edata, _elabel = read_csv(_accel, _gyro, _label, timestep, timeshift)#変更
                    _elabel = np_utils.to_categorical(np.argmax(_elabel[["NO_LABEL", "ROLL", "RUN", "DOOR"]].values, axis = -1), num_classes = 4)

                    if len(_edata) == 0: continue
                    eval = model.evaluate(_edata, _elabel, batch_size = 100, verbose = 1)

                    counts = np.zeros((4, 3))
                    
                    #predicted
                    predicted = model.predict(_edata)
                    #labeldata 読み込み
                    labelcsv = pd.read_csv(_label)
                    print(_label)
                    _label = _label.replace("/home/aj1m0n/jupyter_notebook/Datamining/20180731/cepstrums/", "-").replace(".label", "").replace("/","")
                    
                    #predictedcsv保存
                    predicted_csv = pd.DataFrame(predicted)
                    predicted_csv.columns = ["NO_LABEL", "ROLL", "RUN", "DOOR"]
                    predicted_csv["timestamp"] = labelcsv["timestamp"]
                    #predicted_csv = predicted_csv.set_index("timestamp")
                    #predicted_csv.to_csv("./models/20180806/predict/predicted%s" %_label, index=False)#csv保存
                    predicted_csv.to_csv("%spredicted%s" %(outpredict,_label), index=False)#csv保存
                    global predicted #モジュールのグローバル化
                    
                    #F値の処理
                    try:
                        _tlabel = np.argmax(_elabel, axis = -1)
                        _plabel = np.argmax(predicted, axis = -1)
                        for _t, _p in zip(_tlabel, _plabel):
                            if _t == _p: counts[_p, 0] += 1
                            counts[_p, 1] += 1
                            counts[_t, 2] += 1
                        _macro_fmesure = 0.0
                        _fmesures = []
                        for _i in range(4):
                            if counts[_i, 1] == 0 or counts[_i, 2] == 0: continue
                            _precision = counts[_i, 0] / counts[_i, 1]
                            _recall = counts[_i, 0] / counts[_i, 2]
                            _fmesure = 2 * (_precision * _recall) / (_precision + _recall)
                            _macro_fmesure += _fmesure
                            _fmesures.append(_fmesure)
                        _macro_fmesure /= 4

                        line = [path.basename(_accel), path.basename(_gyro), path.basename(_label), str(_edata.shape[0])
                                , "{:0.8f}".format(eval[0]), "{:0.8f}".format(eval[1]), "{:0.8f}".format(_macro_fmesure)]
                        line += ["{:0.8f}".format(v) for v in _fmesures]
                        f.write(",".join(line))
                        f.write("\n")
                    except:
                        print("ERROR", _edata, predicted)
                        _macro_fmesure = 0.0
                        line = [path.basename(_accel), path.basename(_gyro), path.basename(_label), str(_edata.shape[0])
                                , "{:0.8f}".format(eval[0]), "{:0.8f}".format(eval[1]), "{:0.8f}".format(_macro_fmesure)]
                        f.write(",".join(line))
                        f.write("\n")
    finally:
        session.close()
        print('Goodbye, world!')
