#coding:Shift_JIS


import keras
import cnn_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os
import os.path as path
import mnist_load
import mnist_plot_2d
import csv
from keras.utils import np_utils
import numpy as np



#MNIST読み込み
(x_train, y_train, x_test, y_test) = mnist_load.mnist_2d()


def cnn_ae(encode_dim = 8,latent_dim = 8,epochs = 20):
	#初期定数
	width = 28
	height = 28
	channels = 1
	class_num = 10
	batch_size = 100
	
	#現在時刻の取得
	from datetime import datetime as dt
	tdatetime = dt.now()
	nowtime = tdatetime.strftime('%m%d-%H%M')

		#繰り返し学習回数

	#modelの呼び出し
	(autoencoder,model,classifier) = cnn_model.mnist_cnn_AE_classifer(width,height,channels,class_num,encode_dim,latent_dim)

	autoencoder.summary()
	classifier.summary()
	##### autoencoderの学習 #####
	logdir = path.join('/tmp/mnist_ae/cnn', str(nowtime))
	if not path.isdir(logdir):os.makedirs(logdir);
	tensorboard = keras.callbacks.TensorBoard(log_dir = logdir, batch_size = batch_size,
				histogram_freq = 1,		# モデルの層の活性化ヒストグラムを計算する（エポック中の）頻度．この値を0に設定するとヒストグラムが計算されません．
				write_graph = False,	# TensorBoardのグラフを可視化するか．write_graphがTrueの場合，ログファイルが非常に大きくなることがあります．
				write_grads = False,	# TensorBoardに勾配のヒストグラフを可視化するかどうか．histogram_freqは0より大きくしなければなりません．
				write_images = False,	# TensorfBoardで可視化するモデルの重みを画像として書き出すかどうか．
				)
	history_ae = autoencoder.fit(x_train, x_train,
							epochs = epochs, 
							batch_size = batch_size,
							validation_data = (x_test, x_test),
							shuffle = True
							,callbacks = [tensorboard]
							)
							
							
	model.get_layer(name = "encoder_conv01").trainable = False
	model.get_layer(name = "encoder_pool01").trainable = False
	model.get_layer(name = "encoder_conv02").trainable = False
	model.get_layer(name = "encoder_pool02").trainable = False
	model.get_layer(name = "encoder_conv03").trainable = False
	model.get_layer(name = "encoder_pool03").trainable = False
	
	model.get_layer(name = "decoder_conv01").trainable = False
	model.get_layer(name = "decoder_pool01").trainable = False
	model.get_layer(name = "decoder_conv02").trainable = False
	model.get_layer(name = "decoder_pool02").trainable = False
	model.get_layer(name = "decoder_conv03").trainable = False
	model.get_layer(name = "decoder_pool03").trainable = False
						
	##### classifierの学習 #####
	# encoderのパラメータを固定
	#classifier.get_layer(name = "encoder_conv01").trainable = False
	#classifier.get_layer(name = "encoder_pool01").trainable = False
	#classifier.get_layer(name = "encoder_conv02").trainable = False
	#classifier.get_layer(name = "encoder_pool02").trainable = False
	#classifier.get_layer(name = "encoder_conv03").trainable = False
	#classifier.get_layer(name = "encoder_pool03").trainable = False
	classifier.compile(optimizer = "Adam",metrics=['accuracy'], loss = "mean_squared_error")

	history_class = classifier.fit(x_train, y_train,
									epochs = epochs, 
									batch_size = batch_size,
									validation_data = (x_test, y_test),
									shuffle = True,
									callbacks = [tensorboard],
									verbose=2
									)


	#保存場所の作成
	outdir = "encode_dim_"+ str(encode_dim) + "-latent_dim_" + str(latent_dim)
	outdir = outdir + "--" + nowtime
	outdir = path.join('./cnn_ae_result', outdir)
	if not path.isdir(outdir): os.makedirs(outdir)



	#認識精度の保存
	csvpath = open(path.join(outdir,"loss_result.csv"), "w", newline='')
	write_fp=csv.writer(csvpath)
	loss = history_ae.history['loss']
	val_loss = history_ae.history['val_loss']
	acc = history_class.history['acc']
	val_acc = history_class.history['val_acc']


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
	score = model.evaluate(x_test,x_test,verbose=0)
	f.write("AutoEncoder result")
	f.write("loss="+str(score)+"\n")
	score = classifier.evaluate(x_test,y_test,verbose=0)
	f.write("Classifier result")
	f.write("loss="+str(score[0])+"\n")
	f.write("Acc="+str(score[1])+"\n")

	f.close() # ファイルを閉じる


	#plot関数
	mnist_plot_2d.plot_fig(model,classifier,outdir)
	mnist_plot_2d.rigfht_shift(model,classifier,outdir,shift_px=14)
	mnist_plot_2d.left_paint(model,classifier,outdir,shift_px=14)


	#モデルの保存
	classifier.save(path.join(outdir,"classifier.h5"))  
	autoencoder.save(path.join(outdir,"autoencoder.h5"))  

	print("tensorboard --logdir="+str(logdir))
