import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as path
from scipy.stats import norm
import keras
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import plot_model
import mnist_load
import csv

def plot_fig(model,classifier,outdir):#10個を抜粋表示（変形なし）
	(x_,y_train_label,x_test_origin,y_test_label)=mnist_load.load()
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()

	decoded_imgs = model.predict(x_test)#データをオートエンコーダに
	#decoded_imgs = decoded_imgs.reshape(len(x_test),784)
	decoded_label = classifier.predict(x_test)#データをオートエンコーダに
	decode_difference = np.empty((x_test_origin.shape))
	for i in range(len(x_test)):
		decode_difference[i] = x_test_origin[i] - decoded_imgs[i]


	
	
	plt.figure(figsize=(20, 6))
	n=10
	for i in range(n):
		
		# display original
		ax = plt.subplot(3, n, i+1)
		plt.imshow(x_test[i].reshape(28, 28))
		title = "True:" + str(y_test_label[i])
		plt.title(title)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display reconstruction
		ax = plt.subplot(3, n,i+1+10)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.title(np.argmax(decoded_label[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display difference
		ax = plt.subplot(3, n,i+1+20)
		plt.imshow(decode_difference[i].reshape(28, 28))
		plt.title("Difference")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	
	png = path.join(outdir,"result.png")
	plt.savefig(png)



def rigfht_shift(model,classifier,outdir,shift_px=14):#10個を抜粋表示(左端塗りつぶし・右に移動)
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()
	
	png = path.join(outdir,"result_rightshift.png")
	decoded_imgs = model.predict(x_test)#データをオートエンコーダに
	decoded_label = classifier.predict(x_test)#データをオートエンコーダに
	
	n = 10
	size_h = shift_px*2 +2

	score_all = np.empty((shift_px+1,2))
	score_class = classifier.evaluate(x_test,y_test,verbose=0)
	score_ae = model.evaluate(x_test,x_test,verbose=0)
	score_all[0][0] = score_ae
	score_all[0][1] = score_class[1]

	plt.figure(figsize=(20, 2*size_h))
	for i in range(n):

		# display original
		ax = plt.subplot(size_h, n, i+1)
		plt.imshow(x_test[i].reshape(28, 28))
		title = "True:" + str(y_test_label[i])
		plt.title(title)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display reconstruction
		ax = plt.subplot(size_h, n,i+1+10)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.title(np.argmax(decoded_label[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	for num in range(shift_px):
		(x_train_trans, y_train, x_test_trans, y_test) = mnist_load.mnist_1d_rightshift(num+1)
		
		trans_decoded_imgs = model.predict(x_test_trans)#データをオートエンコーダに入れる
		trans_decoded_label = classifier.predict(x_test_trans)#データをオートエンコーダに入れる
		
		score_class = classifier.evaluate(x_test_trans,y_test,verbose=0)
		score_ae = model.evaluate(x_test_trans,x_test_trans,verbose=0)
		score_all[num+1][0] = score_ae
		score_all[num+1][1] = score_class[1]
		

		for i in range(n):
			# display trans origin
			ax = plt.subplot(size_h, n, i+1+(20*(num+1)))
			plt.imshow(x_test_trans[i].reshape(28, 28))
			title = "True:" + str(y_test_label[i])
			plt.title(title)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			
			# display trans reconstruction
			ax = plt.subplot(size_h, n, i+1+10+(20*(num+1)))
			plt.imshow(trans_decoded_imgs[i].reshape(28, 28))
			plt.title(np.argmax(trans_decoded_label[i]))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

	plt.savefig(png)
	socorefile = open(path.join(outdir,"rightshift_score.csv"), 'w') # 書き込みモードで開く
	socorefile.write("AE_loss,Classifier_acc,shift_px\n")
	for i in range(shift_px+1):
		socorefile.write(str(score_all[i][0]))
		socorefile.write(",")
		socorefile.write(str(score_all[i][1]))
		socorefile.write(",")
		socorefile.write(str(i))
		socorefile.write("\n")
		
	socorefile.close() # ファイルを閉じる

def left_paint(model,classifier,outdir,shift_px=14):#10個を抜粋表示(左端塗りつぶし)
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()
	png = path.join(outdir,"result_leftpaint.png")
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()

	decoded_imgs = model.predict(x_test)#データをオートエンコーダに
	decoded_label = classifier.predict(x_test)#データをオートエンコーダに
	n = 10
	size_h = shift_px*2 +2
	score_all = np.empty((shift_px+1,2))
	score_class = classifier.evaluate(x_test,y_test,verbose=0)
	score_ae = model.evaluate(x_test,x_test,verbose=0)
	score_all[0][0] = score_ae
	score_all[0][1] = score_class[1]

	plt.figure(figsize=(20, 2*size_h))
	for i in range(n):

		# display original
		ax = plt.subplot(size_h, n, i+1)
		plt.imshow(x_test[i].reshape(28, 28))
		title = "True:" + str(y_test_label[i])
		plt.title(title)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display reconstruction
		ax = plt.subplot(size_h, n,i+1+10)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.title(np.argmax(decoded_label[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	for num in range(shift_px):
		(x_test_trans, y_test_) = mnist_load.mnist_1d_leftpaint(num+1)
		
		trans_decoded_imgs = model.predict(x_test_trans)#データをオートエンコーダに入れる
		trans_decoded_label = classifier.predict(x_test_trans)#データをオートエンコーダに入れる
		
		score_class = classifier.evaluate(x_test_trans,y_test,verbose=0)
		score_ae = model.evaluate(x_test_trans,x_test_trans,verbose=0)
		score_all[num+1][0] = score_ae
		score_all[num+1][1] = score_class[1]	

		for i in range(n):
			# display trans origin
			ax = plt.subplot(size_h, n, i+1+(20*(num+1)))
			plt.imshow(x_test_trans[i].reshape(28, 28))
			title = "True:" + str(y_test_label[i])
			plt.title(title)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			
			# display trans reconstruction
			ax = plt.subplot(size_h, n, i+1+10+(20*(num+1)))
			plt.imshow(trans_decoded_imgs[i].reshape(28, 28))
			plt.title(np.argmax(trans_decoded_label[i]))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

	plt.savefig(png)
	
	socorefile = open(path.join(outdir,"leftpaint_score.csv"), 'w') # 書き込みモードで開く
	socorefile.write("AE_loss,Classifier_acc,shift_px\n")
	for i in range(shift_px+1):
		socorefile.write(str(score_all[i][0]))
		socorefile.write(",")
		socorefile.write(str(score_all[i][1]))
		socorefile.write(",")
		socorefile.write(str(i))
		socorefile.write("\n")
		
	socorefile.close() # ファイルを閉じる

def plot_fig_cvae(model,outdir):#10個を抜粋表示（変形なし）
	(x_,y_train_label,x_test_origin,y_test_label)=mnist_load.load()
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()

	decoded_imgs = model.predict([y_test,x_test])#データをオートエンコーダに
	decoded_imgs = decoded_imgs.reshape(len(x_test),784)
	decode_difference = np.empty((x_test_origin.shape))
	for i in range(len(x_test)):
		decode_difference[i] = x_test_origin[i] - decoded_imgs[i]

	#バイナリデータの保存
	binarypath = open(path.join(outdir,"graph.csv"), "w", newline='')
	write_fp= csv.writer(binarypath)
	for num in range(len(x_test)):
		write_fp.writerow(decoded_imgs[num])
	binarypath.close()
	
	
	plt.figure(figsize=(20, 6))
	n=10
	for i in range(n):
		
		# display original
		ax = plt.subplot(3, n, i+1)
		plt.imshow(x_test[i].reshape(28, 28))
		title = "True:" + str(y_test_label[i])
		plt.title(title)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display reconstruction
		ax = plt.subplot(3, n,i+1+10)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display difference
		ax = plt.subplot(3, n,i+1+20)
		plt.imshow(decode_difference[i].reshape(28, 28))
		plt.title("Difference")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	
	png = path.join(outdir,"result.png")
	plt.savefig(png)



def rigfht_shift_cvae(model,outdir,shift_px=14):#10個を抜粋表示(左端塗りつぶし・右に移動)
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()
	
	png = path.join(outdir,"result_rightshift.png")
	decoded_imgs = model.predict([y_test,x_test])#データをオートエンコーダに
	
	n = 10
	size_h = shift_px*2 +2

	score_all = np.empty((shift_px+1,))
	score_ae = model.evaluate([y_test,x_test],x_test,verbose=0)
	score_all[0] = score_ae

	plt.figure(figsize=(20, 2*size_h))
	for i in range(n):

		# display original
		ax = plt.subplot(size_h, n, i+1)
		plt.imshow(x_test[i].reshape(28, 28))
		title = "True:" + str(y_test_label[i])
		plt.title(title)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display reconstruction
		ax = plt.subplot(size_h, n,i+1+10)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	for num in range(shift_px):
		(x_train_trans, y_train, x_test_trans, y_test) = mnist_load.mnist_1d_rightshift(num+1)
		
		trans_decoded_imgs = model.predict([y_test,x_test_trans])#データをオートエンコーダに入れる
		
		score_ae = model.evaluate([y_test,x_test_trans],x_test_trans,verbose=0)
		score_all[num+1] = score_ae
		

		for i in range(n):
			# display trans origin
			ax = plt.subplot(size_h, n, i+1+(20*(num+1)))
			plt.imshow(x_test_trans[i].reshape(28, 28))
			title = "True:" + str(y_test_label[i])
			plt.title(title)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			
			# display trans reconstruction
			ax = plt.subplot(size_h, n, i+1+10+(20*(num+1)))
			plt.imshow(trans_decoded_imgs[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

	plt.savefig(png)
	socorefile = open(path.join(outdir,"rightshift_score.csv"), 'w') # 書き込みモードで開く
	socorefile.write("AE_loss,shift_px\n")
	for i in range(shift_px+1):
		socorefile.write(str(score_all[i]))
		socorefile.write(",")
		socorefile.write(str(i))
		socorefile.write("\n")
		
	socorefile.close() # ファイルを閉じる

def left_paint_cvae(model,outdir,shift_px=14):#10個を抜粋表示(左端塗りつぶし)
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()
	png = path.join(outdir,"result_leftpaint.png")
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()

	decoded_imgs = model.predict([y_test,x_test])#データをオートエンコーダに
	n = 10
	size_h = shift_px*2 +2
	score_all = np.empty((shift_px+1,))
	score_ae = model.evaluate([y_test,x_test],x_test,verbose=0)
	score_all[0] = score_ae

	plt.figure(figsize=(20, 2*size_h))
	for i in range(n):

		# display original
		ax = plt.subplot(size_h, n, i+1)
		plt.imshow(x_test[i].reshape(28, 28))
		title = "True:" + str(y_test_label[i])
		plt.title(title)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
		# display reconstruction
		ax = plt.subplot(size_h, n,i+1+10)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	for num in range(shift_px):
		(x_test_trans, y_) = mnist_load.mnist_1d_leftpaint(num+1)
		
		trans_decoded_imgs = model.predict([y_test,x_test_trans])#データをオートエンコーダに入れる
		
		score_ae = model.evaluate([y_test,x_test_trans],x_test_trans,verbose=0)
		score_all[num+1] = score_ae

		for i in range(n):
			# display trans origin
			ax = plt.subplot(size_h, n, i+1+(20*(num+1)))
			plt.imshow(x_test_trans[i].reshape(28, 28))
			title = "True:" + str(y_test_label[i])
			plt.title(title)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			
			# display trans reconstruction
			ax = plt.subplot(size_h, n, i+1+10+(20*(num+1)))
			plt.imshow(trans_decoded_imgs[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

	plt.savefig(png)
	
	socorefile = open(path.join(outdir,"leftpaint_score.csv"), 'w') # 書き込みモードで開く
	socorefile.write("AE_loss,Classifier_acc,shift_px\n")
	for i in range(shift_px+1):
		socorefile.write(str(score_all[i]))
		socorefile.write(",")
		socorefile.write(str(i))
		socorefile.write("\n")
		
	socorefile.close() # ファイルを閉じる


def label_changed(cvae,outdir):#ラベル変更時の描画

	#MNISTの使用準備
	(x_train, y_train, x_test, y_test) = mnist_load.mnist_1d()
	(x_,y_train_label,y_,y_test_origin)=mnist_load.mnist_1d()
	(x_,y_train_label,y_,y_test_label)=mnist_load.load()

	#ラベル変更時の描画
	n = 10
	h = 10
	label = np.empty(10)
	for i in range(10):
		label[i] = i
	label_byte = keras.utils.to_categorical(label, 10)
	plt.figure(figsize=(20, 2*(h+1)))

	score_all = np.empty((11))
	score = cvae.evaluate([y_test,x_test],x_test,verbose=0)
	score_all[0] = score
	for j in range(n):
		for num in range(len(y_test)):
			y_test[num] = label_byte[j]
		score = cvae.evaluate([y_test,x_test],x_test,verbose=0)
		score_all[j+1] = score

	socorefile = open(path.join(outdir,"label_set_loss.csv"), 'w') # 書き込みモードで開く
	socorefile.write("loss,label\n")
	socorefile.write(str(score_all[0]))
	socorefile.write(",original\n")
	for i in range(n):
		socorefile.write(str(score_all[i+1]))
		socorefile.write(",")
		socorefile.write(str(i))
		socorefile.write("\n")
		
	socorefile.close() # ファイルを閉じる

	for i in range(h):
		ax = plt.subplot(h, n+1,1+(i*11))
		plt.imshow(x_test[i].reshape(28, 28))
		title_name ="T="+str(y_test_label[i])
		plt.title(title_name)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		for j in range(n):
		
			for num in range(len(y_test)):
				y_test[num] = label_byte[j]
			decoded_imgs = cvae.predict([y_test,x_test])
			ax = plt.subplot(h, n+1,j+2+(i*(11)))
			plt.imshow(decoded_imgs[i].reshape(28, 28))
			title_name ="label="+ str(label[j])
			plt.title(title_name)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

	png = path.join(outdir,"result_right.png")
	plt.savefig(png)
