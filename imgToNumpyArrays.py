# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:21:41 2018

@author: s161915
"""

import numpy as np
import pandas as pd
import os
import cv2 as cv
import preprocess


def save_train_data(dir):
    lab = pd.read_csv('TrainLabels.csv')
    filenames = os.listdir(dir)
    imgs = []
    for file in filenames:
        imgs.append(file.split('.')[0])
    
    Y_train = []
        
    for i in imgs:
        
        if lab['image'].isin([i]).any():
            mask = lab['image'].isin([str(i)])
            df = lab[mask]
            Y_train.append(df['level'][df.index[0]])
            
    
    X_train = []
    
    for file in filenames:
        img = cv.imread(dir+file)
        img = preprocess.pp(img)
        X_train.append(img)
        
        
    return(np.savez('trainImages.npz', X_train=X_train, Y_train=Y_train))
    
    
def save_test_data(dir):
    lab = pd.read_csv('testLabels.csv')
    files = os.listdir(dir)
    imgs = []
    for file in files:
        imgs.append(file.split('.')[0])
    
    Y_test = []
        
    for i in imgs:
        
        if lab['image'].isin([i]).any():
            mask = lab['image'].isin([str(i)])
            df = lab[mask]
            Y_test.append(df['level'][df.index[0]])
            
    
    X_test = []
    
    for file in files:
        img = cv.imread(dir+file)
        img = preprocess.pp(img)
        X_test.append(img)
        
        
    return(np.savez('testImages.npz', X_test=X_test, Y_test=Y_test))

