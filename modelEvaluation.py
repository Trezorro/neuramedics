## Checking model evaluation
import os
from keras.models import load_model
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

import random
import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils

import scikitplot as skplt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12,9

import cv2 as cv

os.chdir("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1")
labels = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/augmented_labels.csv")
os.getcwd()
os.chdir("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1")



data_train = np.load("trainDataSmall.npz")
data_test = np.load("testDataSmall.npz")
data_train.files
data_test.files

X_train = data_train["X_train"].astype(np.float32)
Y_train = data_train["Y_train"].astype(np.float32)
X_test = data_test["X_test"].astype(np.float32)
Y_test = data_test["Y_test"].astype(np.float32)



def checkClassImabalanceBinary(labels):
    counter_class0 = 0
    counter_class1 = 0
    for idx, filename in enumerate(labels['image'][85000:]):
        #img = cv.imread(dir+filename+'.jpeg')
        #img = img_to_array(img)
        if labels['level'][idx]==0:
            counter_class0 +=1
        else:
            counter_class1 += 1
    print("Count of healthy: ", counter_class0)
    print("Count of sick: ", counter_class1)

checkClassImabalanceBinary(labels)



### Model evaluation part

model = load_model('/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/retinopathy.hdf5')
#model.evaluate(X_test, Y_test)


predictions = model.predict(X_test)
model.predict(X_test[:1])

def createBinaryY(one_hot_labels):
    lst = [1 if i[0] == 0 else 0 for i in one_hot_labels]
    return keras.utils.to_categorical(lst, num_classes=2)


def get_binary_probabilities(preds):
    probabs = []
    [probabs.append([i[0], i[1]+i[2]+i[3]+i[4]]) for i in preds]
    return np.array(probabs)

binary_probabs = get_binary_probabilities(predictions)
binary_probabs[1]

def get_binary_predictions(preds, threshold):
    binary_preds = []
    [binary_preds.append([1,0]) if i[0]> threshold else binary_preds.append([0,1]) for i in preds]
    return binary_preds

true_binary_Y = get_binary_predictions(Y_test,0.5)
pred_binary_Y = get_binary_predictions(predictions,0.5)


def get_accuracy(preds, labels):
    counter = 0
    for i in range(len(preds)):
        if preds[i]==labels[i]:
            counter +=1
    return counter/len(preds)

get_accuracy(pred_binary_Y, true_binary_Y)

def get_sensitivity(preds,labels):
    tp = 0
    positivies = 0
    for i in range(len(preds)):
        if labels[i] == [0,1]:
            positivies += 1
            if preds[i]==labels[i]:
                tp +=1
    return tp/positivies

get_sensitivity(pred_binary_Y, true_binary_Y)

def get_specificity(preds, labels):
    tn = 0
    negatives = 0
    for i in range(len(preds)):
        if labels[i] == [1,0]:
            negatives += 1
            if preds[i]==labels[i]:
                tn +=1
    return tn/negatives

get_specificity(pred_binary_Y, true_binary_Y)



def plot_roc(predictions, labels): # more: https://scikit-plot.readthedocs.io/en/stable/metrics.html
    probabs = get_binary_probabilities(predictions)
    lables_no_hot = np.array([0 if i[0]==1 else 1 for i in labels])
    skplt.metrics.plot_roc(lables_no_hot, probabs, plot_micro = False, plot_macro=False)
    plt.show()


ax = plot_roc(predictions, true_binary_Y)


### Testing out sampling functions

x_train_small = X_train[:10]
y_train_small = Y_train[:10]

zipped = list(zip(x_train_small, y_train_small))
random.shuffle(zipped)


x_train_small, y_train_small = zip(*zipped)
np.array(x_train_small).shape



def take_random_sample(size, X, Y,seed):
    zipped = list(zip(X,Y))
    random.Random(seed).shuffle(zipped)
    zipped_shuffeled_sampled = zipped[:size]
    sample_x, sample_y = zip(*zipped_shuffeled_sampled)
    return np.array(sample_x), np.array(sample_y)

a,b = take_random_sample(4,x_train_small,y_train_small,7)
a = ['a', 'b', 'c', 'd','e','f','g']
b = [1, 2, 3,4,5,6,7]

a,b = take_random_sample(4,a,b,11)
a
b

### Checking wheteher npz's have similiar properties

jaap128 = np.load("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/testDataResized128.npz")

jX = jaap128['X_test'].astype('float32')
jY = jaap128['Y_test'].astype('float32')

plt.imshow(jX[99])
plt.show()
