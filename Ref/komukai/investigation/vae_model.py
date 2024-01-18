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


#‰Šú•Ï”



#ƒTƒ“ƒvƒŠƒ“ƒOŠÖ”
def sampling(args,**kwargs):
    z_mean, z_log_var = args
    latent_dim = kwargs["latent_dim"]
    epsilon_std = kwargs["epsilon_std"]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


#ƒJƒXƒ^ƒ€‘¹¸ŠÖ”
#end-to-endƒ‚ƒfƒ‹‚ğg—p‚µ‚ÄgÄ\¬€‚Ì˜ah‚ÆgKL”­Uh‚Ì³‘¥‰»‚ğŒP—û‚·‚é
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
        return x_decoded_mean

def model(original_dim = 784,latent_dim = 10,intermediate_dim = 784//4,epsilon_std = 1.0):
	
	with tf.name_scope("VarietionalAutoEncoder") as scope:
		with tf.name_scope("input") as scope:
			input = Input(shape=(original_dim,),name="input")
		
		with tf.name_scope("encoder") as scope:
			encoder01 = Dense(intermediate_dim, activation='relu',name='encoder01')(input)
		with tf.name_scope("mean") as scope:
			Zmean = Dense(latent_dim,name='z_mean')(encoder01)
		with tf.name_scope("var") as scope:
			Zlog_var = Dense(latent_dim,name='z_log_var')(encoder01)
		with tf.name_scope("Z") as scope:
			Zlambda = Lambda(sampling,name='Z', arguments = {"latent_dim": latent_dim, "epsilon_std": epsilon_std})([Zmean,Zlog_var])
		with tf.name_scope("decoder") as scope:
			decoder_h = Dense(intermediate_dim, activation='relu',name='decoder_h')(Zlambda)
			decoder_mean = Dense(original_dim, activation='sigmoid',name='decoder_mean')(decoder_h)
		with tf.name_scope("decode_loss") as scope:
			y = CustomVariationalLayer(original_dim,name="loss")([input, decoder_mean, Zmean, Zlog_var])
		with tf.name_scope("VAEmodel") as scope:
			vae = Model(input, y)
			vae.compile(optimizer='rmsprop', loss="mean_squared_error")
		with tf.name_scope("AutoEncodermodel") as scope:
			autoencoder = Model(input, decoder_mean)
				
	with tf.name_scope("Classifier") as scope:
			
		with tf.name_scope("encoder") as scope:
			Classifier_encoder01 = Dense(intermediate_dim, activation='relu',name='Classifier_encoder01')(decoder_mean)
		with tf.name_scope("mean") as scope:
			Classifier_Zmean = Dense(latent_dim,name='Classifier_z_mean')(Classifier_encoder01)
		with tf.name_scope("var") as scope:
			Classifier_Zlog_var = Dense(latent_dim,name='Classifier_z_log_var')(Classifier_encoder01)
		with tf.name_scope("Z") as scope:
			Classifier_Zlambda = Lambda(sampling,name='Classifier_Z', 
										arguments = {"latent_dim": latent_dim, 
										"epsilon_std": epsilon_std})([Classifier_Zmean,Classifier_Zlog_var])
			
		with tf.name_scope("classifier_input") as scope:
			classifier_input = Dense(latent_dim, activation = "relu", name = "classifier")(Zlambda)
		with tf.name_scope("classifier_output") as scope:
			num_class = 10;
			classifier_output = Dense(num_class, activation = "sigmoid", name = "classifier_output")(classifier_input)
		with tf.name_scope("ClassifierEmodel") as scope:
			classifier = Model(input, classifier_output)



	return (vae,classifier,autoencoder)
	