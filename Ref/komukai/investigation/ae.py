#coding:Shift_JIS

'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as path
from scipy.stats import norm
import keras
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
import ae_model
from keras.utils import plot_model
import csv
import mnist_load
import mnist_plot_1d




#MNISTの使用準備
(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()
	
def autoencoder(intermediate_dim = 196,latent_dim = 40,epochs = 20):
	batch_size = 100

	#modelの読み込み
	(autoencoder,classifier) = ae_model.model(original_dim = 784,
											latent_dim = latent_dim,
											intermediate_dim = intermediate_dim)

	autoencoder.summary()
	classifier.summary()


	#AEの学習
	history_ae = autoencoder.fit(x_train,x_train,
					shuffle=True,
					epochs=epochs,
					batch_size=batch_size,
					validation_data=(x_test, x_test)
					)
			
			
	print(autoencoder.evaluate(x_test,x_test,verbose=0))
	#分類機の学習
	classifier.get_layer(name = "encoder01").trainable = False
	classifier.get_layer(name = "encoder02").trainable = False
	#classifier.get_layer(name = "decoder01").trainable = False
	#classifier.get_layer(name = "decoder02_output").trainable = False
	classifier.compile(optimizer = "Adam", loss = "mean_squared_error", metrics = ["accuracy"])

	history_classifier = classifier.fit(x_train,y_train,
									shuffle=True,
									epochs=epochs,
									batch_size=batch_size,
									validation_data=(x_test, y_test),
									verbose = 2
									)

	print(autoencoder.evaluate(x_test,x_test,verbose=0))
	
	#保存場所の作成
	from datetime import datetime as dt
	tdatetime = dt.now()
	nowtime = tdatetime.strftime('%m%d-%H%M')
	outdir = "intermediate_"+ str(intermediate_dim) + "-latent_" + str(latent_dim)
	outdir = outdir + "--" + nowtime
	outdir = path.join('./fc_ae_result', outdir)
	if not path.isdir(outdir): os.makedirs(outdir)



	#認識精度の保存
	csvpath = open(path.join(outdir,"loss_result.csv"), "w", newline='')
	write_fp=csv.writer(csvpath)
	loss = history_ae.history['loss']
	val_loss = history_ae.history['val_loss']
	acc = history_classifier.history['acc']
	val_acc = history_classifier.history['val_acc']


	head = ["AEloss","AE_val_loss","acc","val_acc"]

	write_fp.writerow(head)  # ヘッダを書き込む
	for i in range(len(loss)):
		num = np.empty(4)
		num[0] = loss[i]
		num[1] = val_loss[i]
		num[2] = acc[i]
		num[3] = val_acc[i]
		write_fp.writerow(num)  # 内容を書き込む
	csvpath.close()


	#実験設定と実験結果スコアの保存
	f = open(path.join(outdir,"result.txt"), 'w') # 書き込みモードで開く
	f.write("AutoEncoder")
	autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
	f.write("Classifieer")
	classifier.summary(print_fn=lambda x: f.write(x + '\n'))

	f.write("Experiment setting\n")
	f.write("epochs="+str(epochs)+"\n")
	f.write("Experiment result\n")
	f.write("loss=mean_squared_error\n")
	
	f.write("AutoEncoder result")
	score = autoencoder.evaluate(x_test,x_test,verbose=0)
	print(score)
	f.write("loss="+str(score)+"\n")
	
	f.write("Classifier result")
	score = classifier.evaluate(x_test,y_test,verbose=0)
	f.write("loss="+str(score[0])+"\n")
	f.write("Acc="+str(score[1])+"\n")

	f.close() # ファイルを閉じる


	#plot関数
	mnist_plot_1d.plot_fig(autoencoder,classifier,outdir)
	mnist_plot_1d.rigfht_shift(autoencoder,classifier,outdir,shift_px=14)
	mnist_plot_1d.left_paint(autoencoder,classifier,outdir,shift_px=14)


	#モデルの保存
	classifier.save(path.join(outdir,"classifier.h5"))  
	autoencoder.save(path.join(outdir,"autoencoder.h5"))  


