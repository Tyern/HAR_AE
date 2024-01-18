import keras
from keras.layers import Dense
from keras.layers import Input
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNGRU
from keras.layers import CuDNNLSTM
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.models import Model
from keras import backend as K
from tensorflow import name_scope as scope

import random as rn
import os
import numpy as np
import tensorflow as tf

#random seed を固定
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
session_conf = tf.ConfigProto(
intra_op_parallelism_threads=1,
inter_op_parallelism_threads=1
)

tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)#固定終わり

def _name(name: str) -> str:
    """tensorflowのname_scopeを「/」を「_」に変換して返す。Kerasの名称では「/」が取り扱えないため。"""
    if name is None or len(name) == 0: return ""
    name = name.replace("/", "_")
    return name[:-1] if name[-1] == "_" else name

def get_model(name=None
              , units=180, batch_size=20, time_step=1
              , feature_dim=180, label_size=3
              , loss=u"categorical_crossentropy", optimizer=u"adam", metrics=[u"accuracy"], layer="LSTM"):
        input = Input(shape = (time_step, feature_dim), name = "Input")

        # Recurrent層
        if layer == "GRU":
            recurrent_layer = GRU(units, name = "GRU", implementation = 2)(input)
        elif layer == "CuDNNGRU":
            recurrent_layer = CuDNNGRU(units, name = "CuDNNGRU")(input)
        elif layer == "LSTM":
            recurrent_layer = LSTM(units, name = "LSTM", implementation = 2)(input)
        elif layer == "CuDNNLSTM":
            recurrent_layer = CuDNNLSTM(units, name = "CuDNNLSTM")(input)
        else:
            raise ValueError("Unknown Recurrent Layer type: {}".format(layer))
        dense1 = Dense(units, kernel_initializer = "normal", activation = "relu", name = "Dense1")(recurrent_layer)
        dense2 = Dense(units, kernel_initializer = "normal", activation = "relu", name = "Dense2")(dense1)
        dense3 = Dense(label_size, kernel_initializer = "normal", activation = "relu", name = "Dense3")(dense2)
        out = Dense(label_size, kernel_initializer = "normal", activation = "softmax", name = "out")(dense3)

        model = Model(name = name, inputs = [input], outputs = [out])
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        return model

def get_ConvLSTM_model(time_step:int=1, feature_size:int=128, sensor_types:int=2, axis_size:int=3
                       # ConvLSTM2D パラメータ
                       , filters:int=180, kernel_size=(2, 16), strides=(1, 8), padding:str="same" , activation="tanh", recurrent_activation="hard_sigmoid"
                       # Dense パラメータ
                       , units=[180, 90, 45], dense_activation="relu"
                       , label_size=4):

    with scope("input") as name: input = Input(shape = (time_step, sensor_types, feature_size, axis_size), name = _name(name))
    with scope("ConvLSTM") as name: x = ConvLSTM2D(1, kernel_size, strides = strides, padding = padding, activation = activation, recurrent_activation = recurrent_activation, name = _name(name))(input)
    # with scope("Conv") as name: x = Conv2D(1, (kernel_size[0], 1), padding =
    # padding, activation = activation, name = _name(name))(x)
    with scope("Flatten") as name: x = Flatten(name = _name(name))(x)
    with scope("Dnese") as name:
        for index, unit in enumerate(units):
            with scope("%d" % index) as name: x = Dense(unit, activation = dense_activation, name = _name(name))(x)
    with scope("output") as name: x = Dense(label_size, activation = "softmax", name=_name(name))(x)

    with scope("model") as name: model = Model(name = _name(name), inputs = input, outputs = x)
    return model

def get_Conv_model(time_step:int=1, feature_size:int=128, channel_size:int=6
                   , filters=[32, 16], kernel_size=(1, 16), strides=(1, 8)
                   , units=[180, 90, 45], label_size:int=4
                   , conv_activation="tanh", recurrent_activation="hard_sigmoid", dense_activation="relu"):
    with scope("input") as name: input = Input(shape=(time_step, feature_size, channel_size), name = _name(name))
    x = input
    _feture_size = feature_size
    for _idx, _filter in enumerate(filters):
        with scope("Conv_%d" % _idx) as name: x = Conv2D(_filter, kernel_size, strides = strides, padding = "same", activation = conv_activation, name = _name(name))(x)
        with scope("MaxPooling_%d" % _idx) as name: x = MaxPooling2D(pool_size = (1, 2), name = _name(name))(x)
        _feture_size //= 2
    with scope("Conv_%d" % (_idx + 1)) as name: x = Conv2D(1, kernel_size, strides = strides, padding = "same", activation = conv_activation, name = _name(name))(x)
    with scope("Reshape") as name: x = Reshape((time_step, -1), name = _name(name))(x)
    with scope("LSTM") as name: x = LSTM(units[-1], activation = conv_activation, recurrent_activation = recurrent_activation, name = _name(name))(x)
    with scope("Dnese") as name:
        for index, unit in enumerate(units):
            with scope("%d" % index) as name: x = Dense(unit, activation = dense_activation, name = _name(name))(x)
    with scope("output") as name: x = Dense(label_size, activation = "softmax", name=_name(name))(x)

    with scope("model") as name: model = Model(name = _name(name), inputs = input, outputs = x)
    return model

