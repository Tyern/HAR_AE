import pandas as pd
import numpy as np

from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
from keras.optimizers import Adam
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os.path
from os import path
import random as rn
import glob

import create_model
from create_model import read_csv
from create_model import read_datas

def relearning(f_cep: str, f_model: str, f_weight: str, car_ids: list, eval_car_ids: list, outeval: str, outmodel: str, outweights: str, outpredict: str, batch_size: int, epochs: int, units: int, timestep: int, timeshift: int):
    #pra
    #batch_size = 100
    #epochs = 40
    #units = 180
    #timestep = 3
    #timeshift = 1

    #ファイルの読み込み
    #f_cep = "./cepstrums/"
    #eval_car_ids = ["001"]
    #f_model = "./models/20180806/model/"
    #model_filename = 'model005_010_011_013.model'
    #f_weithts = "./models/20180806/weight"
    #weights_filename = "weight005_010_011_013.h5"
    #outeval = "./models/20180807/act_eval/test.csv"
    #outmodel = "./models/20180807/model/test.model"
    #outweights = "./models/20180807/weight/test.h5"
    #outpredict = "./"

    ##学習
    for car_id in car_ids:
        model_filename = [s for s in glob.glob(f_model + "*") if not  car_id in s]
        #model_filename = glob.glob(f_model + "*")
        
        print(model_filename)
        for mfn in model_filename:
            mfn = str(mfn).replace("[","").replace("]","").replace("'","")
            days = glob.glob(f_cep + "%s/*.label.csv" %car_id)#日付データ読み込み
            print(days)
            for day_path in days:
                print(day_path)
                day = day_path.replace("%s"%f_cep,"").replace("%s/"%car_id,"").replace(".vague","").replace(".label.csv","")
                print(day)
                os.environ['PYTHONHASHSEED'] = '0'
                np.random.seed(1)
                rn.seed(1)
                session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
                )
                tf.set_random_seed(1)
                sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
                KTF.set_session(sess)#固定終わり
                try:
                    #曖昧データ読み込み
                    xdata, xlabel, ydata, ylabel = create_model.read_datas(f_cep,[day_path],timestep,timeshift)#car_id を日付を与える形に治す○
                    #model読み込み
                    json_string = str(mfn)
                    print(json_string)
                    #model = model_from_json(json_string)#json形式の場合    
                    model = load_model(json_string)
                    model.summary()
                    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])

                    weight_filename = mfn.replace(f_model,"").replace(".model",".h5").replace("model","weight")
                    model.load_weights(os.path.join(f_weight,weight_filename))#weightファイル読み込み
                    model.fit(xdata, xlabel, batch_size = 100, epochs = epochs, callbacks = [],
                            validation_data = [ydata, ylabel],#変えるべき
                            verbose = 2)
                    if os.path.exists(outmodel + str(day)) == False:os.mkdir(outmodel + str(day))#フォルダ作成
                    model.save(outmodel + "%s/" %day + "re-" + mfn.replace(f_model,""))#変えるべき
                    if os.path.exists(outweights + str(day)) == False:os.mkdir(outweights + str(day))#フォルダ作成
                    model.save_weights(outweights + "%s/" %day + "re-" + weight_filename)#変えるべき

                    if os.path.exists(outeval + str(day)) == False:os.mkdir(outeval + str(day))#フォルダ作成    
                    with open(outeval + "%s/"%day + car_id + mfn.replace(f_model,"").replace(".model",".csv").replace("model","eval"), "wt") as f:#変えるべき
                        print(outeval + "%s/"%day + car_id + mfn.replace(f_model,"").replace(".model",".csv").replace("model","eval"))
                        _headers = ["accel", "gyro", "label","data_counts", "loss", "accuracy", "macro-fmesure"]
                        _headers += ["NO_LABEL", "ROLL", "RUN", "DOOR"]
                        f.write(",".join(_headers))
                        f.write("\n")
                        #if args.eval_car_ids is None or len(args.eval_car_ids) == 0: file_ptns = [path.join(args.f_cep, "*/*.label.csv")]
                        if eval_car_ids is None or len(eval_car_ids) == 0: file_ptns = [os.path.join(f_cep, "*/*.label.csv")]
                        #else: file_ptns = [path.join(args.f_cep, "%03d/*.label.csv" % i) for i in args.eval_car_ids]
                        else: file_ptns = [os.path.join("./cepstrums/", "%s/*.label.csv" % i) for i in eval_car_ids]
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
                                _label = _label.replace("/home/aj1m0n/jupyter_notebook/Datamining/20180731/cepstrums/", "-").replace(".label", "").replace("/","").replace("cepstrums","-")
                                
                                #predictedcsv保存
                                predicted_csv = pd.DataFrame(predicted)
                                predicted_csv.columns = ["NO_LABEL", "ROLL", "RUN", "DOOR"]
                                predicted_csv["timestamp"] = labelcsv["timestamp"]
                                #predicted_csv = predicted_csv.set_index("timestamp")
                                #predicted_csv.to_csv("./models/20180806/predict/predicted%s" %_label, index=False)#csv保存
                                if os.path.exists(outpredict + str(day)) == False:os.mkdir(outpredict + str(day))#フォルダ作成
                                predicted_csv.to_csv("%s%s/predicted%s" %(outpredict,day,_label), index=False)#csv保存
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
                    sess.close()
                    print('Goodbye, world!')

def Decrelearning(f_cep: str, f_model: str, f_weight: str, car_ids: list, eval_car_ids: list, outeval: str, outmodel: str, outweights: str, outpredict: str, batch_size: int, epochs: int, units: int, timestep: int, timeshift: int):
    #pra
    #batch_size = 100
    #epochs = 40
    #units = 180
    #timestep = 3
    #timeshift = 1

    #ファイルの読み込み
    #f_cep = "./cepstrums/"
    #eval_car_ids = ["001"]
    #f_model = "./models/20180806/model/"
    #model_filename = 'model005_010_011_013.model'
    #f_weithts = "./models/20180806/weight"
    #weights_filename = "weight005_010_011_013.h5"
    #outeval = "./models/20180807/act_eval/test.csv"
    #outmodel = "./models/20180807/model/test.model"
    #outweights = "./models/20180807/weight/test.h5"
    #outpredict = "./"

    ##学習
    for car_id in car_ids:
        model_filename = [s for s in glob.glob(f_model + "*") if not  car_id in s]
        #model_filename = glob.glob(f_model + "*")
        
        print(model_filename)
        for mfn in model_filename:
            mfn = str(mfn).replace("[","").replace("]","").replace("'","")
            days = glob.glob(f_cep + "%s/*.label.csv" %car_id)#日付データ読み込み
            print(days)
            for day_path in days:
                print(day_path)
                day = day_path.replace("%s"%f_cep,"").replace("%s/"%car_id,"").replace(".vague","").replace(".label.csv","")
                print(day)
                os.environ['PYTHONHASHSEED'] = '0'
                np.random.seed(1)
                rn.seed(1)
                session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
                )
                tf.set_random_seed(1)
                sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
                KTF.set_session(sess)#固定終わり
                try:
                    #曖昧データ読み込み
                    xdata, xlabel, ydata, ylabel = create_model.read_datas(f_cep,[day_path],timestep,timeshift)#car_id を日付を与える形に治す○
                    #model読み込み
                    json_string = str(mfn)
                    print(json_string)
                    #model = model_from_json(json_string)#json形式の場合    
                    model = load_model(json_string)
                    model.summary()
                    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])

                    weight_filename = mfn.replace(f_model,"").replace(".model",".h5").replace("model","weight")
                    model.load_weights(os.path.join(f_weight,weight_filename))#weightファイル読み込み
                    model.fit(xdata, xlabel, batch_size = 100, epochs = epochs, callbacks = [],
                            validation_data = [ydata, ylabel],#変えるべき
                            verbose = 2)
                    if os.path.exists(outmodel + str(day)) == False:os.mkdir(outmodel + str(day))#フォルダ作成
                    model.save(outmodel + "%s/" %day + "re-" + mfn.replace(f_model,""))#変えるべき
                    if os.path.exists(outweights + str(day)) == False:os.mkdir(outweights + str(day))#フォルダ作成
                    model.save_weights(outweights + "%s/" %day + "re-" + weight_filename)#変えるべき

                    if os.path.exists(outeval + str(day)) == False:os.mkdir(outeval + str(day))#フォルダ作成    
                    with open(outeval + "%s/"%day + car_id + mfn.replace(f_model,"").replace(".model",".csv").replace("model","eval"), "wt") as f:#変えるべき
                        print(outeval + "%s/"%day + car_id + mfn.replace(f_model,"").replace(".model",".csv").replace("model","eval"))
                        _headers = ["accel", "gyro", "label","data_counts", "loss", "accuracy", "macro-fmesure"]
                        _headers += ["NO_LABEL", "ROLL", "RUN", "DOOR"]
                        f.write(",".join(_headers))
                        f.write("\n")
                        #if args.eval_car_ids is None or len(args.eval_car_ids) == 0: file_ptns = [path.join(args.f_cep, "*/*.label.csv")]
                        if eval_car_ids is None or len(eval_car_ids) == 0: file_ptns = [os.path.join(f_cep, "*/*.label.csv")]
                        #else: file_ptns = [path.join(args.f_cep, "%03d/*.label.csv" % i) for i in args.eval_car_ids]
                        else: file_ptns = [os.path.join("./cepstrums/", "%s/*.label.csv" % i) for i in eval_car_ids]
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
                                _label = _label.replace("/home/aj1m0n/jupyter_notebook/Datamining/20180731/cepstrums/", "-").replace(".label", "").replace("/","").replace("cepstrums","-")
                                
                                #predictedcsv保存
                                predicted_csv = pd.DataFrame(predicted)
                                predicted_csv.columns = ["NO_LABEL", "ROLL", "RUN", "DOOR"]
                                predicted_csv["timestamp"] = labelcsv["timestamp"]
                                #predicted_csv = predicted_csv.set_index("timestamp")
                                #predicted_csv.to_csv("./models/20180806/predict/predicted%s" %_label, index=False)#csv保存
                                if os.path.exists(outpredict + str(day)) == False:os.mkdir(outpredict + str(day))#フォルダ作成
                                predicted_csv.to_csv("%s%s/predicted%s" %(outpredict,day,_label), index=False)#csv保存
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
                    sess.close()
                    print('Goodbye, world!')

