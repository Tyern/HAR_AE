#coding:Shift_JIS
from keras.layers import Dense
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

def mnist_cnn_AE_classifer(width = 28,height = 28,channels = 1,class_num = 10,encode_dim = 8,latent_dim = 8):

	optimizer = "Adam"
	#loss = "mean_squared_error"
	loss = "binary_crossentropy"
	#入力
	input = Input(shape = (width, height, channels,),name="input")
	input_dec = Input(shape = (width // 2, height // 2, latent_dim), name = "input_decoder")
	#AE出力
	output = Conv2D(1, (3, 3), activation = "sigmoid", padding = "same", name = "decoder_output")

	#エンコーダーのひな形作成
	encoder_conv01 = Conv2D(encode_dim, (3, 3), activation = "relu", padding = "same", name = "encoder_conv01")
	encoder_pool01 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "encoder_pool01")
	encoder_conv02 = Conv2D(latent_dim, (3, 3), activation = "relu", padding = "same", name = "encoder_conv02")
	encoder_pool02 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "encoder_pool02")
	encoder_conv03 = Conv2D(latent_dim, (3, 3), activation = "relu", padding = "same", name = "encoder_conv03")
	encoder_pool03 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "encoder_pool03")

	#デコーダのひな形作成
	decoder_conv01 = Conv2D(latent_dim, (3, 3), activation = "relu", padding = "same", name = "decoder_conv01")
	decoder_pool01 = UpSampling2D(size = (2, 2), name = "decoder_pool01")
	decoder_conv02 = Conv2D(encode_dim, (3, 3), activation = "relu", padding = "same", name = "decoder_conv02")
	decoder_pool02 = UpSampling2D(size = (2, 2), name = "decoder_pool02")
	decoder_conv03 = Conv2D(encode_dim, (3, 3), activation = "relu",  name = "decoder_conv03")
	decoder_pool03 = UpSampling2D(size = (2, 2), name = "decoder_pool03")

	#分類機のひな形作成
	classifier_encoder_conv01 = Conv2D(encode_dim, (3, 3), activation = "relu", padding = "same", name = "classifier_encoder_conv01")
	classifier_encoder_pool01 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "classifier_encoder_pool01")
	classifier_encoder_conv02 = Conv2D(latent_dim, (3, 3), activation = "relu", padding = "same", name = "classifier_encoder_conv02")
	classifier_encoder_pool02 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "classifier_encoder_pool02")
	classifier_encoder_conv03 = Conv2D(latent_dim, (3, 3), activation = "relu", padding = "same", name = "classifier_encoder_conv03")
	classifier_encoder_pool03 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "classifier_encoder_pool03")
	
	classifier_conv01 = Conv2D(class_num, (3, 3), activation = "relu", padding = "same", name = "classifier_conv01")
	classifier_pool01 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "classifier_pool01")
	classifier_conv02 = Conv2D(class_num, (3, 3), activation = "relu", padding = "same", name = "classifier_conv02")
	classifier_pool02 = MaxPooling2D(pool_size = (2, 2), padding = "same", name = "classifier_pool02")
	flatten = Flatten()
	classifier = Dense(class_num, activation = "sigmoid", name = "classifier")
	#classifier01 = Dense((width // 4) * (height // 4) * latent_dim, activation = "relu", name = "classifier01")
	#classifier02 = Dense(32, activation = "relu", name = "classifier02")
	#classifier03 = Dense(class_num, activation = "sigmoid", name = "classifier03")

	#オートエンコーダの生成
	x = encoder_conv01(input) 
	x = encoder_pool01(x) 
	x = encoder_conv02(x)
	x = encoder_pool02(x)
	x = encoder_conv03(x)
	x = encoder_pool03(x)
	encoder_tensor = x # エンコード結果
	x = decoder_conv01(x) 
	x = decoder_pool01(x) 
	x = decoder_conv02(x) 
	x = decoder_pool02(x)
	x = decoder_conv03(x) 
	x = decoder_pool03(x) 
	x = output(x)

	autoencoder = Model(inputs = input, outputs = x)
	autoencoder.compile(optimizer = optimizer, loss = loss)
	
	model = Model(inputs = input, outputs = x)
	model.compile(optimizer = optimizer, loss = "mean_squared_error")
	
	#分類機の生成
	#入力はエンコード結果
	y = classifier_encoder_conv01(input)
	y = classifier_encoder_pool01(y)
	y = classifier_encoder_conv02(y)
	y = classifier_encoder_pool02(y)
	y = classifier_encoder_conv03(y)
	y = classifier_encoder_pool03(y)
	
	y = classifier_conv01(y)
	y = classifier_pool01(y)
	y = classifier_conv02(y)
	y = classifier_pool02(y)
	y = flatten(y)
	y = classifier(y) 
	#x = flatten(encoder_tensor)
	#x = classifier01(x) 
	#x = classifier02(x) 
	#x = classifier03(x)
	classifier = Model(inputs = input, outputs = y)
	#classifier.compile(optimizer = optimizer, metrics=['accuracy'], loss =  "mean_squared_error")

	return autoencoder,model,classifier
