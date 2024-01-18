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
import vae_model
from keras.utils import plot_model
import csv
import mnist_load
import mnist_plot_1d




#MNISTの使用準備
(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()


def create_metadatas(imgpath, metapath, xtest, ytest, num_class = 10, predicted = None):
    from PIL import Image;
    if not path.isfile(imgpath):
        img_array = xtest.reshape(100, 100, 28, 28);
        img_array_flatten = np.concatenate([np.concatenate([x for x in row], axis = 1) for row in img_array]);
        img = Image.fromarray(np.uint8(255 * (1.0 - img_array_flatten)));
        img.save(imgpath);
    if predicted is None:
        with open(metapath, "wt") as f:
            f.write("Index\tLabel\n");
            for index, label in enumerate(ytest): f.write("{}\t{}\n".format(index, label));
    else:
        # メタデータに正解ラベル、正解したかどうか、およびpredictの確率を追加
        predicted_labels = np.argmax(predicted, axis = 1);
        with open(metapath, "wt") as f:
            f.write("Index\tLabel\tPredict\tProb\tHit\n");
            for index, label in enumerate(ytest):
                pred = predicted_labels[index];
                prob = predicted[index, pred];
                hit = 1 if pred == min(label, num_class) else 0;
                f.write("{}\t{}\t{}\t{}\t{}\n".format(index, label, pred, prob, hit));

def varietionnal_autoencoder(intermediate_dim = 196,latent_dim = 40,epochs = 20):
	batch_size = 100

	#modelの読み込み
	(vae,classifier,autoencoder) = vae_model.model(original_dim = 784,latent_dim = latent_dim,intermediate_dim = intermediate_dim,epsilon_std = 1.0)

	autoencoder.summary()
	classifier.summary()
	logdir = path.join('/tmp/mnist_vae', "vae")
	import shutil
	if path.isdir(logdir): shutil.rmtree(logdir)#前のログデータが残ってたら削除
	os.makedirs(logdir);
	imgfile = "images.jpg"; 
	metafile = "metadata.tsv";
	from TensorResponseBoard import TensorResponseBoard;
	tensorboard = TensorResponseBoard(
				log_dir = logdir, 
				batch_size = batch_size,
				histogram_freq = 1,		# モデルの層の活性化ヒストグラムを計算する（エポック中の）頻度．この値を0に設定するとヒストグラムが計算されません．
				write_graph = True,	# TensorBoardのグラフを可視化するか．write_graphがTrueの場合，ログファイルが非常に大きくなることがあります．
				write_grads = True,	# TensorBoardに勾配のヒストグラフを可視化するかどうか．histogram_freqは0より大きくしなければなりません．
				write_images = 1	# TensorfBoardで可視化するモデルの重みを画像として書き出すかどうか．
				, embeddings_freq = 1, embeddings_layer_names = ["z_mean"]
				, embeddings_metadata = metafile, img_path = imgfile, img_size = [28, 28], val_size = len(x_test)
				)
	# create image, metadata
	imgpath = path.join(logdir, imgfile);
	metapath = path.join(logdir, metafile);
	create_metadatas(imgpath, metapath, x_test, y_test, predicted = None);

	#VAEの学習
	history_ae = vae.fit(x_train,x_train,
					shuffle=True,
					epochs=epochs,
					batch_size=batch_size,
					validation_data=(x_test, x_test)
					#,callbacks = [tensorboard]
					)
			
			

	#VAE(loss=平均二乗誤差版)の重みづけ
	autoencoder.compile(optimizer = "Adam", loss = "mean_squared_error")
	autoencoder.get_layer(name = "encoder01").trainable = False
	autoencoder.get_layer(name = "z_mean").trainable = False
	autoencoder.get_layer(name = "z_log_var").trainable = False
	autoencoder.get_layer(name = "Z").trainable = False
	autoencoder.get_layer(name = "decoder_h").trainable = False
	autoencoder.get_layer(name = "decoder_mean").trainable = False
	#VAEの学習
	autoencoder.fit(x_train,x_train,
					shuffle=True,
					epochs=epochs,
					initial_epoch=epochs, 
					batch_size=batch_size,
					validation_data=(x_test, x_test)
					#,callbacks = [tensorboard]
					)

	#分類機の学習
	classifier.compile(optimizer = "Adam", loss = "mean_squared_error", metrics = ["accuracy"]);
	classifier.get_layer(name = "encoder01").trainable = False
	classifier.get_layer(name = "z_mean").trainable = False
	classifier.get_layer(name = "z_log_var").trainable = False
	classifier.get_layer(name = "Z").trainable = False
	#classifier.get_layer(name = "decoder_h").trainable = False
	#classifier.get_layer(name = "decoder_mean").trainable = False

	history_classifier = classifier.fit(x_train,y_train,
									shuffle=True,
									epochs=epochs,
									batch_size=batch_size,
									validation_data=(x_test, y_test),
									verbose = 2
									)

	#保存場所の作成
	from datetime import datetime as dt
	tdatetime = dt.now()
	nowtime = tdatetime.strftime('%m%d-%H%M')
	outdir = "intermediate_"+ str(intermediate_dim) + "-latent_" + str(latent_dim)
	outdir = outdir + "--" + nowtime
	outdir = path.join('./fc_vae_result', outdir)
	if not path.isdir(outdir): os.makedirs(outdir)



	#認識精度の保存
	csvpath = open(path.join(outdir,"loss_result.csv"), "w", newline='')
	write_fp=csv.writer(csvpath)
	loss = history_ae.history['loss']
	val_loss = history_ae.history['val_loss']
	acc = history_classifier.history['acc']
	val_acc = history_classifier.history['val_acc']


	head = ["VAEloss","VAE_val_loss","acc","val_acc"]

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
	f.write("VarietionalAutoEncoder（学習用）")
	autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
	f.write("VarietionalAutoEncoder（通常用）")
	vae.summary(print_fn=lambda x: f.write(x + '\n'))
	f.write("Classifieer")
	classifier.summary(print_fn=lambda x: f.write(x + '\n'))

	f.write("Experiment setting\n")
	f.write("epochs="+str(epochs)+"\n")
	f.write("Experiment result\n")
	f.write("loss=mean_squared_error\n")
	score = autoencoder.evaluate(x_test,x_test,verbose=0)
	f.write("AutoEncoder result")
	f.write("loss="+str(score)+"\n")
	score = classifier.evaluate(x_test,y_test,verbose=0)
	f.write("Classifier result")
	f.write("loss="+str(score[0])+"\n")
	f.write("Acc="+str(score[1])+"\n")

	f.close() # ファイルを閉じる


	#plot関数
	mnist_plot_1d.plot_fig(autoencoder,classifier,outdir)
	mnist_plot_1d.rigfht_shift(autoencoder,classifier,outdir,shift_px=14)
	mnist_plot_1d.left_paint(autoencoder,classifier,outdir,shift_px=14)


	#モデルの保存
	vae.save(path.join(outdir,"vae.h5"))  
	classifier.save(path.join(outdir,"classifier.h5"))  
	autoencoder.save(path.join(outdir,"encoder.h5"))  

	print("tensorboard --logdir="+str(logdir))

