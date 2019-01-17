## Checking model evaluation
import os
from keras.models import load_model
import pandas as pd
import numpy as np

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
import itertools
from sklearn.metrics import confusion_matrix
import scipy.misc

os.chdir("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/")
labels = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/csv_files/augmented_labels.csv")
os.getcwd()
os.chdir("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/npz_files/")
### Looking at medium npz rich

input_dir = "/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/seeImages"
testMedium = np.load("testDataMediumTrenary.npz")
X_medium = testMedium['X_test'].astype(np.float32)
X_medium.shape
def saveImages(input_dir, X):
    for i in range (0,X.shape[0]):
        scipy.misc.imsave("outphoto{0}.jpg".format(i),X[i])

#saveImages(input_dir,X_medium)
plt.imshow(X_medium[800])
plt.show()
###

#Opening the default 5 class X and Y
data_train = np.load("trainDataSmall.npz")
data_test = np.load("testDataSmall.npz")
data_train.files
data_test.files

X_train = data_train["X_train"].astype(np.float32)
Y_train = data_train["Y_train"].astype(np.float32)
X_test = data_test["X_test"].astype(np.float32)
Y_test = data_test["Y_test"].astype(np.float32)

### Checking the ternary model on augmented data with presumably 76% accuracy

#Loading the preprocessed npz

data_test_augment = np.load("testDataMediumTrenaryAugmentIndep.npz")
X_test_augment = data_test_augment["X_test"].astype(np.float32)
Y_test_augment = data_test_augment["Y_test"].astype(np.float32)

model3 = load_model('/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/best_so_far.hdf5')
eval_model_augment = model3.evaluate(X_test_augment, Y_test_augment)
predictions3 = model3.predict(X_test_augment)

#Loading the preprocessed  npz fixed

data_test_augment_fixed = np.load("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/npz_files/testDataMediumTrenaryAugmentIndep_fixed.npz")
X_test_augment_fixed = data_test_augment_fixed["X_test"].astype(np.float32)
Y_test_augment_fixed = data_test_augment_fixed["Y_test"].astype(np.float32)

model3_aug_fixed = load_model('/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/Models/maren_augment_indep_fixed_79.hdf5')
eval_model_augment_fixed= model3_aug_fixed.evaluate(X_test_augment_fixed, Y_test_augment_fixed)
eval_model_augment_fixed
predictions3_fixed = model3_aug_fixed.predict(X_test_augment_fixed)


### Checking the ternary model on not-augmented data with presumably 61% accuracy
data_test_no_augment = np.load("testDataMediumTrenary.npz")
X_test_no_augment = data_test_no_augment["X_test"].astype(np.float32)
Y_test_no_augment = data_test_no_augment["Y_test"].astype(np.float32)

model3NoAug = load_model('/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/best_so_farNoPre.hdf5')
eval_model_no_augment = model3NoAug.evaluate(X_test_no_augment, Y_test_no_augment)
eval_model_no_augment
predictions3_no_augment= model3NoAug.predict(X_test_no_augment)


### Generating confusion matrix

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


### Model evaluation part
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



### Model without pre processing:

#Confusion matrix

class_names = ["Healthy", "Not so healthy", "Sick", "Super Sick", "Death"]
cnf_matrix = confusion_matrix(Y_test_no_augment.argmax(axis = 1), predictions3_no_augment.argmax(axis = 1))
plot_confusion_matrix(cnf_matrix, classes=class_names[:3], normalize=False,
                      title='Normalized confusion matrix')
plt.savefig("confusion_61%.png")
plt.show()

# Evaluation measueres

true_binary_Y = get_binary_predictions(Y_test_no_augment,0.5)
pred_binary_Y = get_binary_predictions(predictions3_no_augment,0.74)
get_accuracy(pred_binary_Y, true_binary_Y)
#get_best_accuracy(Y_test_no_augment, predictions3_no_augment)
get_sensitivity(pred_binary_Y, true_binary_Y)
get_specificity(pred_binary_Y, true_binary_Y)

# Roc curve

plot_roc_ternary(predictions3_no_augment, Y_test_no_augment)
plt.savefig("RocModelNoPre61.png")
plt.show()

### Model with pre processing:
data_dep= np.load("testDataMediumTrenaryAugment.npz")
X_test_augment_dep = data_dep["X_test"].astype(np.float32)
Y_test_augment_dep = data_dep["Y_test"].astype(np.float32)
eval_model_augment_dep = model3.evaluate(X_test_augment_dep, Y_test_augment_dep)
eval_model_augment_dep
predictions3_augment_dep = model3.predict(X_test_augment_dep)

# Confusion Matrix
class_names = ["Healthy", "Not so healthy", "Sick", "Super Sick", "Death"]
cnf_matrix = confusion_matrix(Y_test_augment.argmax(axis = 1), predictions3.argmax(axis = 1))
plot_confusion_matrix(cnf_matrix, classes=class_names[:3], normalize=False,
                      title='Normalized confusion matrix')
plt.savefig("confusion_too_good.png")
plt.show()

#Evaluation measures

true_binary_Y = get_binary_predictions(Y_test_augment,0.5)
pred_binary_Y = get_binary_predictions(predictions3,0.5)
get_accuracy(pred_binary_Y, true_binary_Y)
get_best_accuracy(Y_test_augment, predictions3)
get_sensitivity(pred_binary_Y, true_binary_Y)
get_specificity(pred_binary_Y, true_binary_Y)

#Roc curve

plot_roc_ternary(predictions3, Y_test_augment)
plt.savefig("RocModelPre76.png")
plt.show()

### Model with pre processing: fixed
data_test_augment_fixed = np.load("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/npz_files/testDataMediumTrenaryAugmentIndep_fixed.npz")
X_test_augment_fixed = data_test_augment_fixed["X_test"].astype(np.float32)
Y_test_augment_fixed = data_test_augment_fixed["Y_test"].astype(np.float32)

model3_aug_fixed = load_model('/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/Models/maren_augment_indep_fixed_79.hdf5')
eval_model_augment_fixed= model3_aug_fixed.evaluate(X_test_augment_fixed, Y_test_augment_fixed)
eval_model_augment_fixed
predictions3_fixed = model3_aug_fixed.predict(X_test_augment_fixed)

# Confusion Matrix
class_names = ["Healthy", "Moderately sick", "Severely sick"]
cnf_matrix = confusion_matrix(Y_test_augment_fixed.argmax(axis = 1), predictions3_fixed.argmax(axis = 1))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Normalized confusion matrix')
plt.savefig("fixed_cnf_79.png")
plt.show()

#Evaluation measures

true_binary_Y = get_binary_predictions(Y_test_augment_fixed,0.5)
pred_binary_Y = get_binary_predictions(predictions3_fixed,0.2)
get_accuracy(pred_binary_Y, true_binary_Y)
#get_best_accuracy(Y_test_augment_fixed, predictions3_fixed)
get_sensitivity(pred_binary_Y, true_binary_Y)
get_specificity(pred_binary_Y, true_binary_Y)

#Roc curve

plot_roc_ternary(predictions3_fixed, Y_test_augment_fixed)
plt.savefig("RocModelPreFix79.png")
plt.show()


## Eval on test set from "test images"
data_test_augment_fixed_new = np.load("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/npz_files/testDataMediumTrenaryAugmentIndep_fixed_new.npz")
X_test_augment_fixed_new = data_test_augment_fixed_new["X_test"].astype(np.float32)
Y_test_augment_fixed_new = data_test_augment_fixed_new["Y_test"].astype(np.float32)

#model3_aug_fixed = load_model('/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/Models/maren_augment_indep_fixed_79.hdf5')
eval_model_augment_fixed_new = model3_aug_fixed.evaluate(X_test_augment_fixed_new, Y_test_augment_fixed_new)
eval_model_augment_fixed_new
### Testing out sampling functions

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

### Writing function to take number of samples of each class

def createTernaryY(one_hot_labels):
    lst = [0 if i[0] == 1 else 1 if (i[1] == 1 or i[2]==1) else 2 for i in one_hot_labels]
    return keras.utils.to_categorical(lst, num_classes=3)

def take_balanced(CLASS_SIZE, X, Y, sample_num):
    sample_indices = {}
    counter = 0
    for i in range(0, CLASS_SIZE): # initialize a dictionary with list of indices as values
        sample_indices["class{0}".format(i)] = []
    while (not all(len(value) == sample_num for value in sample_indices.values())) and (counter != (len(Y))):
        class_to_put = int(np.where([Y[counter] == 1])[1]) #establishes the class of the instance
        if len(sample_indices['class{0}'.format(class_to_put)]) < sample_num:
            sample_indices['class{0}'.format(class_to_put)].append(counter)
        counter +=1
    indices = [item for sublist in sample_indices.values() for item in sublist]
    X_sub = np.array([X[k] for k in indices])
    Y_sub = np.array([Y[z] for z in indices])
    return X_sub, Y_sub
ternary_Y = createTernaryY(Y_test)
ternary_Y[:3]
a, b = take_balanced(3, X_test, ternary_Y , 6)
a.shape
b
