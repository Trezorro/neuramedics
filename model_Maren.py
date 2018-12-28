"""Implements the Keras Sequential model. """

import itertools
import multiprocessing.pool
import threading
from functools import partial

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU 
from keras.layers.normalization import BatchNormalization
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.backend import relu, sigmoid
import numpy as np
import time

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
import os

INIT_CWD = os.getcwd()
DATA_PATH = r'../TUE/20184102/datasets/'

def model_fn(labels_dim):
    """Create a Keras Sequential model with layers."""

    model = models.Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(128, 128, 3)))
    model.add(Conv2D(64, (7, 7), strides=(2, 2)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Dropout(0.4))
    model.add(Dense(labels_dim, activation='softmax'))
    compile_model(model)
    return model
	
	

def compile_model(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def read_train_data():
    start_time = time.time()
    print("Start Read Train Data")
    data = np.load(DATA_PATH + "trainDataSmall.npz")
    print("Train data read --- %s seconds ---" % (time.time() - start_time))
    print(data)
    X_train = data["X_train"] # TODO
    Y_train = data["Y_train"]
    print("Training - Total examples per class", np.sum(Y_train, axis=0))
    return [X_train, Y_train]


def createTernaryY(one_hot_labels): 
    lst = [0 if i[0] == 1 else 1 if (i[1] == 1 or i[2]==1) else 2 for i in one_hot_labels] 
    return keras.utils.to_categorical(lst, num_classes=3) 

def take_balanced(CLASS_SIZE, X, Y, sample_num):
    sample_indices = {}
    counter = 0
    for i in range(0, CLASS_SIZE): 
    # initialize a dictionary with list of indices as values 
		 sample_indices = {}
    while (not all(len(value) == sample_num for value in sample_indices.values())) and (counter != (len(Y))):
		 class_to_put = int(np.where([Y[counter] == 1])[1]) #establishes the class of the instance 
		 if len(sample_indices['class{0}'.format(class_to_put)]) < sample_num:
			 sample_indices['class{0}'.format(class_to_put)].append(counter)
		 counter +=1 
    indices = [item for sublist in sample_indices.values() for item in sublist]
    X_sub = np.array([X[k] for k in indices]) 
    Y_sub = np.array([Y[z] for z in indices]) 
    return X_sub, Y_sub 
	

trinary_Y = createTernaryY(Y_test) 
trinary_Y[:3]


a, b = take_balanced(3, X_test, trinary_Y , 6)


a.shape 
b




def read_test_data():
    start_time = time.time()
    print("Start Read Test Data")
    data = np.load(DATA_PATH + "testDataSmall.npz")
    print("Test data read --- %s seconds ---" % (time.time() - start_time))
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    print("Testing - Total examples per class", np.sum(Y_test, axis=0))
    return [X_test, Y_test]
