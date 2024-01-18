#cording:Shift_JIS
import os;
import os.path as path;
import sys;
sys.path.append(path.dirname(__file__));

from keras.layers import Input, Dense, Lambda, Layer, concatenate;
from keras.layers.core import Flatten, Reshape;
from keras.models import Model;
from keras import backend as K;
from keras import metrics;
import tensorflow as tf;

import numpy as np;


#初期変数



#サンプリング関数
def sampling(args,**kwargs):
    z_mean, z_log_var = args
    latent_dim = kwargs["latent_dim"]
    epsilon_std = kwargs["epsilon_std"]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


#カスタム損失関数
#end-to-endモデルを使用して“再構成項の和”と“KL発散”の正則化を訓練する
class CustomVariationalLayer(Layer):
    def __init__(self, original_dim,**kwargs):
        self.is_placeholder = True
        self.original_dim = original_dim
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):
        xent_loss = np.prod(self.original_dim) * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

def model(original_dim = 784,latent_dim = 10,intermediate_dim = 784//4,epsilon_std = 1.0):
	
	with tf.name_scope("VarietionalAutoEncoder") as scope:
		with tf.name_scope("input") as scope:
			label_input = Input(shape = (10, ), name = "label_input")
			input = Input(shape=(original_dim,),name="input")

		with tf.name_scope("merge_encode") as scope:#エンコーダにラベルを渡す
			merge_encode = concatenate([label_input, input], name = "merge_encode")
		with tf.name_scope("encoder") as scope:
			encoder01 = Dense(intermediate_dim, activation='relu',name='encoder01')(merge_encode)
			
		with tf.name_scope("mean") as scope:
			Zmean = Dense(latent_dim,name='z_mean')(encoder01)
		with tf.name_scope("var") as scope:
			Zlog_var = Dense(latent_dim,name='z_log_var')(encoder01)
		with tf.name_scope("Z") as scope:
			Zlambda = Lambda(sampling,name='Z', arguments = {"latent_dim": latent_dim, "epsilon_std": epsilon_std})([Zmean,Zlog_var])
		
		with tf.name_scope("merge_decode") as scope:# エンコード結果zに対し、ラベル情報をマージする
			merge_decode = concatenate([label_input, Zlambda], name = "merge_decode");
		
		with tf.name_scope("decoder") as scope:
			decoder_h = Dense(intermediate_dim, activation='relu',name='decoder_h')(merge_decode)
			decoder_mean = Dense(original_dim, activation='sigmoid',name='decoder_mean')(decoder_h)
		with tf.name_scope("decode_loss") as scope:
			y = CustomVariationalLayer(original_dim,name="loss")([input, decoder_mean, Zmean, Zlog_var])
		with tf.name_scope("VAEmodel") as scope:
			model = Model([label_input,input], y)
			model.compile(optimizer='rmsprop', loss="binary_crossentropy")
			autoencoder = Model([label_input,input], decoder_mean)
				



	return (model,autoencoder)
	