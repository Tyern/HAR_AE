import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

from keras import metrics;
import tensorflow as tf;


def sampling(args,**kwargs):
    z_mean, z_log_var = args
    latent_dim = kwargs["latent_dim"]
    epsilon_std = kwargs["epsilon_std"]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, img_rows, img_cols, **kwargs):
        self.is_placeholder = True
        self.img_rows = img_rows
        self.img_cols = img_cols
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = self.img_rows * self.img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x_decoded_mean_squash


def model(img_rows, img_cols, img_chns,batch_size,latent_dim = 2,intermediate_dim = 128,
									epsilon_std = 1.0,num_conv = 3,filters = 64,num_class=10):

	with tf.name_scope("VarietionalAutoEncoder") as scope:
		loss = "mean_squared_error"
		original_img_size = (img_rows, img_cols, img_chns)
		img_chns = 1
		with tf.name_scope("input") as scope:
			input = Input(shape=original_img_size,name="input")
		
		with tf.name_scope("encoder") as scope:
			encoder_conv01 = Conv2D(filters, (3, 3), activation = "relu", padding = "same", name = "encoder_conv01")(input)
			encoder_pool01 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "encoder_pool01")(encoder_conv01)
			encoder_conv02 = Conv2D(filters, (3, 3), activation = "relu", padding = "same", name = "encoder_conv02")(encoder_pool01)
			encoder_pool02 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "encoder_pool02")(encoder_conv02)
			
		with tf.name_scope("hidden_reshape") as scope:
			#flat = Flatten(name="flatten")(encoder_pool02)
			#hidden = Dense(intermediate_dim, activation='relu',name="hidden")(flat)
			flat = Flatten(name="flatten")(encoder_pool02)
			hidden = flat
		with tf.name_scope("mean") as scope:
			z_mean = Dense(latent_dim,name="Z_mean")(hidden)
			
		with tf.name_scope("log_var") as scope:
			z_log_var = Dense(latent_dim,name="Z_log_var")(hidden)

		with tf.name_scope("Z") as scope:
			output_shape = (batch_size, 7, 7, filters)
			z = Lambda(sampling, output_shape=(latent_dim,),name="Z",
					arguments = {"latent_dim": latent_dim, "epsilon_std": epsilon_std})([z_mean, z_log_var])

		with tf.name_scope("decoder") as scope:
		# we instantiate these layers separately so as to reuse them later
			#decoder_hid = Dense(intermediate_dim, activation='relu',name="DecoderHid")(z)
			decoder_hid = z
			decoder_upsample = Dense(filters * 7 * 7, activation='relu',name="DecoderUpsample")(decoder_hid)

			decoder_reshape = Reshape(output_shape[1:],name="reshape")(decoder_upsample)

			decoder_conv02 = Conv2D(filters, (3, 3), activation = "relu", padding = "same", name = "decoder_conv03")(decoder_reshape)
			decoder_pool02 = UpSampling2D(size = (2, 2), name = "decoder_pool02")(decoder_conv02)
			decoder_conv01 = Conv2D(filters, (3, 3), activation = "relu", padding = "same", name = "decoder_conv02")(decoder_pool02)
			decoder_pool01 = UpSampling2D(size = (2, 2), name = "decoder_pool01")(decoder_conv01)
			
			decoder_mean_squash = Conv2D(img_chns,kernel_size=2,padding='same',
											activation='sigmoid',name="DecoderConv04")(decoder_pool01)

		with tf.name_scope("DecodedLoss") as scope:
			y = CustomVariationalLayer(img_rows, img_cols,name="Loss")([input, decoder_mean_squash, z_mean, z_log_var])

		with tf.name_scope("VAEModel") as scope:
			vae = Model(input, y)
			vae.compile(optimizer='rmsprop', loss=None)
			autoencoder = Model(input, decoder_mean_squash)
		with tf.name_scope("Classifier") as scope:
			with tf.name_scope("classifier_input") as scope:
				classifier_input = Dense(latent_dim, activation = "relu", name = "classifier")(z)
			with tf.name_scope("classifier_output") as scope:

				classifier_output = Dense(num_class, activation = "sigmoid", name = "classifier_output")(classifier_input)
			with tf.name_scope("ClassifierEmodel") as scope:
				classifier = Model(input, classifier_output)

	return (vae,autoencoder,classifier)
