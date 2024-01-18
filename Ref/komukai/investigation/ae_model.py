#cording:Shift_JIS
import os;
import os.path as path;
import sys;
sys.path.append(path.dirname(__file__));

from keras.layers import Input, Dense, Lambda, Layer;
from keras.layers.core import Flatten, Reshape;
from keras.models import Model;
from keras import backend as K;
from keras import metrics;
import tensorflow as tf;

import numpy as np;


#�����ϐ�



def model(original_dim = 784,latent_dim = 10,intermediate_dim = 784//4):
	class_num = 10
	input = Input(shape=(original_dim,),name="input")
	#�G���R�[�_�ЂȌ`
	encoder01 = Dense(intermediate_dim, activation='relu',name='encoder01')
	encoder02 = Dense(latent_dim, activation='relu',name='encoder02')
	#�f�R�[�_�ЂȌ`
	decoder01 = Dense(intermediate_dim, activation='relu',name='decoder01')
	decoder02 = Dense(original_dim, activation='relu',name='decoder02_output')
	#���ފ�ЂȌ`
	classifier_encoder01 = Dense(intermediate_dim, activation='relu',name='classifier_encoder01')
	classifier_encoder02 = Dense(latent_dim, activation='relu',name='classifier_encoder02')
	classifier = Dense(class_num, activation = "sigmoid", name = "classifier")
	
	x = encoder01(input)
	x = encoder02(x)
	encoder_tensor = x # �G���R�[�h����
	x = decoder01(x)
	output = decoder02(x)
	
	autoencoder = Model(input, output)
	autoencoder.compile(optimizer='Adam', loss="mean_squared_error")
	
	#y = classifier_encoder01(output)
	#y = classifier_encoder02(y)
	y = classifier(encoder_tensor)
	classifier = Model(input, y)


	return (autoencoder,classifier)
	