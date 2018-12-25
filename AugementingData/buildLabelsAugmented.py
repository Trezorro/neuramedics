import numpy as np
import cv2 as cv
import os
import time
import pandas as pd

dir = '/mnt/server-home/dc_group08/data/preprocessingBlazej/train_images/'
labels = pd.read_csv('/mnt/server-home/TUE/20184102/datasets/trainLabels.csv')


def strip_paths(dir):
    return [file.strip(".jpeg") for file in os.listdir(dir)]

paths_entire_dataset = strip_paths(dir)

def hyphen_split(a):
    if a.count("_") == 1:
        return a.split("_")[0]
    else:
        return "_".join(a.split("_", 2)[:2])


def create_labels(paths_entire_dataset, labels):
    valid_imgs = {}
    not_valid_imgs = {}
    for file in paths_entire_dataset:
        if labels['image'].isin([hyphen_split(file)]).any():
            valid_imgs[file] = int( labels[labels['image'] == hyphen_split(file)]['level'])
        else:
            not_valid_imgs[file] = 311298
    return valid_imgs, not_valid_imgs

print("Functions have been excecuted. Starting the work on executing create_labels")

valid_imgs, not_valid_imgs = create_labels(paths_entire_dataset, labels)

labels_augmented = pd.DataFrame.from_dict(valid_imgs, orient = "index").reset_index()
labels_augmented.columns = ['image', 'level']
print("labels_augmented succesfully transformed to data frame")

not_valid_imgs = pd.DataFrame.from_dict(not_valid_imgs, orient = "index").reset_index()
print("not_valid_imgs succesfully transformed to data frame")
#not_valid_imgs.columns = ['image', 'myBirthday']


os.chdir("/mnt/server-home/dc_group08/data/preprocessingBlazej/")


labels_augmented.to_csv("labels_augmented.csv")
print("labels_augmented file saved at:", os.getcwd())
not_valid_imgs.to_csv("not_valid_imgs.csv")
print("not_valid_imgs file saved at:", os.getcwd()
