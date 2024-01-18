#coding:Shift_JIS

import keras
from keras.datasets import mnist
import numpy as np
num_classes = 10

def load():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(60000, 784) # 2�����z���1�����ɕϊ�
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')/255	# int�^��float32�^�ɕϊ�
	x_test = x_test.astype('float32')/255	# [0-255]�̒l��[0.0-1.0]�ɕϊ�
	
	return x_train,y_train,x_test,y_test
	
def mnist_1d():
	(x_train, y_train, x_test, y_test) = load()
	#�N���X���x�����o�C�i���\�L�֕ύX
	num_classes = 10
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train,y_train,x_test,y_test
	
def mnist_2d():
	(x_train, y_train, x_test, y_test) = load()
	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  
	#�N���X���x�����o�C�i���\�L�֕ύX
	num_classes = 10
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train,y_train,x_test,y_test

def mnist_1d_rightshift(shiftpx = 1):
	(x_train, y_train, x_test, y_test) = mnist_2d_rightshift(shiftpx)
	x_train = x_train.reshape(60000, 784) # 2�����z���1�����ɕϊ�
	x_test = x_test.reshape(10000, 784)
	
	return x_train,y_train,x_test,y_test
	
	
def mnist_2d_rightshift(shiftpx = 1):
	(x_train, y_train, x_test, y_test) = load()

	####�E������Npx���炵�A���[Npx��0�i���j�Ŗ��߂�
	x_train_2d = np.reshape(x_train, (len(x_train), 28, 28, 1))  #�I���W�i���摜
	x_train_2d_trans =  np.empty((len(x_train_2d),28,28,1)) #�ό`���ꂽ�摜�p�A�����T�C�Y�̋�W�����擾
	x_test_2d = np.reshape(x_test, (len(x_test), 28, 28, 1))  #�I���W�i���摜
	x_test_2d_trans =  np.empty((len(x_test_2d),28,28,1)) #�ό`���ꂽ�摜�p�A�����T�C�Y�̋�W�����擾
	
	for num in range(len(x_train_2d)):
		x_train_2d_trans[num] = np.roll(x_train_2d[num], shiftpx)
	for num in range(len(x_train_2d)):
		for i in range(28):
			x_train_2d_trans[num][i][0][0] = 0
	
	for num in range(len(x_test_2d)):
		x_test_2d_trans[num] = np.roll(x_test_2d[num], shiftpx)
	for num in range(len(x_test_2d)):
		for i in range(28):
			for j in range(shiftpx):
				x_test_2d_trans[num][i][j][0] = 0

	#�N���X���x�����o�C�i���\�L�֕ύX
	num_classes = 10
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	return x_train_2d_trans,y_train,x_test_2d_trans,y_test
	
	
def mnist_1d_leftpaint(px = 1):
	(x_test_1d_paint,y_test) = mnist_2d_leftpaint(px)
	x_test_1d_paint = x_test_1d_paint.reshape(10000, 784)
	return (x_test_1d_paint,y_test)
	
	
def mnist_2d_leftpaint(px = 1):
	(x_train, y_train, x_test, y_test) = mnist_2d()

	####�E������Npx���炵�A���[Npx��0�i���j�Ŗ��߂�
	
	x_test_2d_paint =  np.empty((len(x_test),28,28,1)) #�ό`���ꂽ�摜�p�A�����T�C�Y�̋�W�����擾
	
	x_test_2d_paint = x_test
	
	for num in range(len(x_test)):
		for i in range(28):
			for j in range(px):
				x_test_2d_paint[num][i][j][0] = 0

	#�N���X���x�����o�C�i���\�L�֕ύX
	num_classes = 10
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	return x_test_2d_paint,y_test