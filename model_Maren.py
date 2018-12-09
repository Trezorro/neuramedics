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
from keras.layers import LeakyReLU #import to make use of leaky relu activation
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
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2,2)
                     activation='relu',
                     input_shape=(128, 128, 3)))
	model.add(MaxPooling2D(pool_size=(3, 3), stride=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), stride=(2,2)))
	model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
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


def read_test_data():
    start_time = time.time()
    print("Start Read Test Data")
    data = np.load(DATA_PATH + "testDataSmall.npz")
    print("Test data read --- %s seconds ---" % (time.time() - start_time))
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    print("Testing - Total examples per class", np.sum(Y_test, axis=0))
    return [X_test, Y_test]
