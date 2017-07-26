from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='device=gpu1,lib.cnmem=0.25,mode=FAST_RUN,floatX=float32,optimizer=fast_compile'
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

X = np.random.rand(1000,784) # 1000个784维的数据



# Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded1 = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded1 = Dense(784, activation='sigmoid')(encoded1)
# model
autoencoder = Model(input=input_img, output=decoded1)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X, X) # 全部选取为默认参数



# Regularized Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# add weight decay to parameters
encoded = Dense(encoding_dim, activation='relu'
	,kernel_regularizer=regularizers.l2(0.01))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
# model
regularized_autoencoder = Model(input=input_img, output=decoded)
regularized_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
regularized_autoencoder.fit(X, X) # 全部选取为默认参数



# Sparse Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# add sparse constraint to activation code
encoded = Dense(encoding_dim, activation='relu'
	,activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
# model
sparse_autoencoder = Model(input=input_img, output=decoded)
sparse_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
sparse_autoencoder.fit(X, X) # 全部选取为默认参数



# Denoising Auto-encoder
# dimension of code
encoding_dim = 100
input_img = Input(shape=(784,))
# corrupted input
corrupted_input = GaussianNoise(0.2)(input_img)
encoded = Dense(encoding_dim, activation='relu')(corrupted_input)
decoded = Dense(784, activation='sigmoid')(encoded)
# model
denoising_autoencoder = Model(input=input_img, output=decoded)
denoising_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
denoising_autoencoder.fit(X, X) # 全部选取为默认参数
