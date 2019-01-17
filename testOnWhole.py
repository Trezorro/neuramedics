import os
from keras.models import load_model
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#import seaborn as sns

import random
import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils

import scikitplot as skplt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12,9


data = np.load("/mnt/server-home/dc_group08/data/npz/TEST2.npz")
X_test = data['X_test']
Y_test = data['Y_test']

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

model = load_model("/mnt/server-home/dc_group08/blazej/jobs/maren_augment_indep_fixed_79.hdf5")

print("Starting evaluating model...")
eval_model_augment = model.evaluate(X_test, Y_test)
print("Model evaluation result:", eval_model_augment)

print("Starting generating predictions...")
predictions3 = model.predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def createBinaryY(one_hot_labels):
    lst = [1 if i[0] == 0 else 0 for i in one_hot_labels]
    return keras.utils.to_categorical(lst, num_classes=2)


def createTrinaryY(one_hot_labels):
    lst = [0 if i[0] == 1 else 1 if (i[1] == 1 or i[2]==1) else 2 for i in one_hot_labels]
    return keras.utils.to_categorical(lst, num_classes=3)

def get_binary_probabilities(preds):
    probabs = []
    [probabs.append( [ i[0], (i[1]+i[2])]) for i in preds]
    return np.array(probabs)


def get_ternary_probabilities(preds):
    probabs = []
    [probabs.append([i[0], i[1]+i[2], i[3]+i[4]]) for i in preds]
    return np.array(probabs)

def get_binary_predictions(preds, threshold):
    preds = get_binary_probabilities(preds)
    binary_preds = []
    [binary_preds.append([1,0]) if i[0]> threshold else binary_preds.append([0,1]) for i in preds]
    return binary_preds

def get_accuracy(preds, labels):
    counter = 0
    for i in range(len(preds)):
        if preds[i]==labels[i]:
            counter +=1
    return counter/len(preds)



def get_best_accuracy(true, pred):
    best = {"Accuracy":0, "Threshold": 0}
    acc = 0
    for i in np.arange(0,1,0.01):
        true_binary_Y = get_binary_predictions(true,i)
        pred_binary_Y = get_binary_predictions(pred,i)
        acc = get_accuracy(pred_binary_Y, true_binary_Y)
        if acc > best["Accuracy"]:
            best["Accuracy"] = acc
            best["Threshold"] = i
    return best

def get_sensitivity(preds,labels):
    tp = 0
    positivies = 0
    for i in range(len(preds)):
        if labels[i] == [0,1]:
            positivies += 1
            if preds[i]==labels[i]:
                tp +=1
    return tp/positivies
def get_specificity(preds, labels):
    tn = 0
    negatives = 0
    for i in range(len(preds)):
        if labels[i] == [1,0]:
            negatives += 1
            if preds[i]==labels[i]:
                tn +=1
    return tn/negatives

def plot_roc_binary(predictions, labels): # more: https://scikit-plot.readthedocs.io/en/stable/metrics.html
    probabs = get_binary_probabilities(predictions)
    lables_no_hot = np.array([0 if i[0]==1 else 1 for i in labels])
    skplt.metrics.plot_roc(lables_no_hot, probabs, plot_micro = False, plot_macro=False)
    plt.show()

#ax = plot_roc_binary(predictions3_no_augment, true_binary_Y)

def plot_roc_ternary(predictions, labels): # more: https://scikit-plot.readthedocs.io/en/stable/metrics.html
    probabs = predictions
    lables_no_hot = np.array([0 if i[0]==1 else 1 if i[1] ==1 else 2 for i in labels])
    skplt.metrics.plot_roc(lables_no_hot, probabs, plot_micro = False, plot_macro=False)

# Confusion Matrix
class_names = ["Healthy", "Moderately sick", "Severely sick"]
cnf_matrix = confusion_matrix(Y_test.argmax(axis = 1), predictions.argmax(axis = 1))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Confusion matrix')
plt.savefig("/mnt/server-home/dc_group08/blazej/confusion_too_good.png")
plt.show()

#Evaluation measures
print("Best accuracy for decsion threshold of:", get_best_accuracy(Y_test, predictions))
for i in np.arange(0,1.01, step=0.01):
    print("Decision threshold = ", i)
    true_binary_Y = get_binary_predictions(Y_test,0.5)
    pred_binary_Y = get_binary_predictions(predictions,i)
    print("Accuracy = ", get_accuracy(pred_binary_Y, true_binary_Y), "Sensitivity =", get_sensitivity(pred_binary_Y, true_binary_Y), "Specificity = ", get_specificity(pred_binary_Y, true_binary_Y) )
