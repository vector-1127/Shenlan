# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 01:53:44 2016

@author: test
"""

from __future__ import absolute_import
from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='device=gpu7,lib.cnmem=0.95,mode=FAST_RUN,floatX=float32,optimizer=fast_compile'
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense,MaxPooling2D,Convolution2D,Highway
from keras.layers import Dropout,Flatten,Input,BatchNormalization
from keras import backend as K
import theano.tensor as T
from keras.utils import np_utils
from keras.engine.topology import Layer,InputSpec
from keras.regularizers import l2, activity_l2,activity_l1
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import model_from_json

batch_size = 128
nb_classes = 10
nb_epoch = 2
img_channels,img_rows,img_cols = 1,28,28
nb_layer = 20



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, img_channels,img_rows,img_cols)
X_test = X_test.reshape(10000, img_channels,img_rows,img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

inp_img = Input(shape=(img_channels,img_rows,img_cols,))
x = Flatten()(inp_img)
x = Dense(32,activation = 'relu')(x)
for _ in range(nb_layer):
    x = Highway(activation = 'relu')(x)
y = Dense(nb_classes,activation = 'softmax')(x)

classifier = Model(input = inp_img,output = y)
classifier.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])
classifier.fit(X_train,Y_train,nb_epoch = 100,batch_size = batch_size,verbose = 1)

