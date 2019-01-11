import pandas as pd
import numpy as np
import os

all_labels = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/csv_files/labels_augmented_fixed.csv", usecols=['image', 'level'])
original_labels = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/csv_files/trainLabels.csv")

all_labels = all_labels.sample(frac =1, random_state = 1998)
def convert_ternary(row):
    if (row == 1) | (row == 2):
        return 1
    elif (row == 3) | (row == 4):
        return 2
    else: return 0

all_labels.groupby(by=['level']).aggregate("count")
original_labels.groupby(by=['level']).aggregate("count")
all_labels['level'] = all_labels.apply(lambda row: convert_ternary(row['level']), axis = 1)
original_labels['level'] = original_labels.apply(lambda row: convert_ternary(row['level']), axis = 1)
all_labels.groupby(by=['level']).aggregate("count")
original_labels.groupby(by=['level']).aggregate("count")

def hyphen_split(a):
    if a.count("_") == 1:
        return a.split("_")[0]
    else:
        return "_".join(a.split("_", 2)[:2])

uniqe_ids = set(all_labels['image'].apply(hyphen_split))
all_labels[:1]
def get_train_test(size_train, size_test, original_labels):
    ## We are bounded by the size of class 3
    train0 = original_labels[original_labels['level']==0][(size_test*3)+1: (size_test+size_train)*3+1]['image'] # *3 because in augmented we double healthy and 6x sick
    train1 = original_labels[original_labels['level']==1][size_test: size_test+size_train]['image']
    train2 = original_labels[original_labels['level']==2][size_test: size_test+size_train]['image']

    test0 = original_labels[original_labels['level']==0][:(size_test*3)+1]['image']
    test1 = original_labels[original_labels['level']==1][:size_test]['image']
    test2 = original_labels[original_labels['level']==2][:size_test]['image']

    train_images = list(train0) + list(train1) + list(train2)
    test_images = list(test0) + list(test1) + list(test2)

    return train_images, test_images

train_images, test_images = get_train_test(1281, 300, original_labels)

train_images_csv = all_labels[all_labels['image'].str.match('|'.join(train_images))]
test_images_csv = all_labels[all_labels['image'].str.match('|'.join(test_images))]

train_images_csv.groupby(by=['level']).aggregate("count")
test_images_csv.groupby(by=['level']).aggregate("count")

train_images_csv.to_csv("train_images_augment_fixed.csv")
test_images_csv.to_csv("test_images_augment_fixed.csv")

### Checking the independence

train_images = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/train_images_augment_fixed.csv", usecols=['image', 'level'])
test_images = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/test_images_augment_fixed.csv", usecols=['image', 'level'])

uniqe_ids_train = set(train_images['image'].apply(hyphen_split))
uniqe_ids_test = set(test_images['image'].apply(hyphen_split))
any(i in uniqe_ids_train for i in uniqe_ids_test)
