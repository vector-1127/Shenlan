# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:17:15 2016

@author: test
"""


from __future__ import absolute_import
from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile,lib.cnmem=0.45'
import numpy as np

from keras.datasets import cifar10,mnist
from keras.models import Sequential, Model
from keras.layers import Dense,MaxPooling2D,Convolution2D,Highway,AveragePooling2D,Activation
from keras.layers import Dropout,Flatten,Input,BatchNormalization,AveragePooling3D,Reshape
from keras import backend as K
import theano.tensor as T
from keras.utils import np_utils
from keras.engine.topology import Layer,InputSpec
from keras.regularizers import l2, activity_l2,activity_l1
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import model_from_json
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.layers.noise import GaussianNoise
sgd = SGD(lr=0.002, decay=0.00001, momentum=0.9, nesterov=True)


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.18,
    height_shift_range=0.18,
    channel_shift_range=0.1,
    horizontal_flip=True,
    rescale=0.95,
    zoom_range=[0.85,1.15]
)



batch_size = 100
nb_classes = 10
nb_epoch = 2
img_channels,img_rows,img_cols = 3,32,32
dim = img_channels*img_rows*img_cols
nb_layer = 10
aug = 16
n_fintune = [300,1000,5000,10000,30000,50000]
nb_iteration = 2
aug_iteration = 32

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(50000, img_channels,img_rows,img_cols)
X_test = X_test.reshape(10000, img_channels,img_rows,img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

inp_img = Input(shape=(img_channels,img_rows,img_cols))
x = Reshape((img_channels*img_rows*img_cols,))(inp_img)
for _ in range(nb_layer):
    x = Dense(dim)(x)
    x = BatchNormalization(mode=2)(x)
    x = Activation('relu')(x)
encode = Dense(dim,activation = 'relu')(x)

decoded = Dense(dim)(encode)
decoded = BatchNormalization(mode=2)(decoded)
decoded = Activation('relu')(decoded)
for _ in range(nb_layer-1):
    decoded = Dense(dim)(decoded)
    decoded = BatchNormalization(mode=2)(decoded)
    decoded = Activation('relu')(decoded)
decoded = Dense(img_channels*img_rows*img_cols,activation = 'sigmoid')(encode)
y = Reshape((img_channels,img_rows,img_cols))(decoded)

pre_train_model = Model(inp_img,encode)
pre_train_model.compile(loss = 'mse',optimizer = 'rmsprop')

pre_train = Model(inp_img,y)
pre_train.compile(loss = 'mse',optimizer = 'rmsprop')
pre_train.fit(X_train,X_train, batch_size=256,nb_epoch = 1,verbose = 2)

pre_train_string = pre_train_model.to_json()
pre_train_model.save_weights('pre_train_AE{}.h5'.format(nb_layer))

res = []
for nb_fintune in n_fintune:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(50000, img_channels,img_rows,img_cols)
    X_test = X_test.reshape(10000, img_channels,img_rows,img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    inp_img = Input(shape=(img_channels,img_rows,img_cols))
    x = Reshape((img_channels*img_rows*img_cols,))(inp_img)
    for _ in range(nb_layer):
        x = Dense(dim)(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
    encode = Dense(dim,activation = 'relu')(x)
    y = Dense(nb_classes,activation = 'softmax')(encode)
    
    pre_train_model = Model(inp_img,encode)
    pre_train_model.compile(loss = 'mse',optimizer = 'rmsprop')
    
    pre_train_model.load_weights('pre_train_AE{}.h5'.format(nb_layer))
    
    classifier = Model(input = inp_img,output = y)
    classifier.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])
    
    classifier.fit_generator(datagen.flow(X_train[0:nb_fintune],Y_train[0:nb_fintune], batch_size=batch_size),validation_data = [X_test,Y_test],nb_epoch = 150,samples_per_epoch=nb_fintune,verbose = 2)
    
    score = classifier.evaluate(X_test,Y_test,batch_size = batch_size,verbose=0)
    print(score)