# -*- coding: utf-8 -*-
"""
Created on Wed Nov  28 15:17:38 2018

@author: s161915
"""

##importing libraries that are used
import numpy as np
import cv2 as cv
import os
import time
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

##defining directory of folder containing images
dirin = '/mnt/server-home/TUE/20184102/datasets/test/'
dirout = '/mnt/server-home/dc_group08/data/preprocessing/test_images/'

##function that rescales an image based on a factor value (v). e.g. if an image is 200x300 and v is 1/2
##the output will be an image which is 100x150
def rescale(img, v):
    height, width = img.shape[:2]
    nh = int(height *v)
    nw = int(width *v)
    res = cv.resize(img,(nw, nh), interpolation = cv.INTER_LINEAR)
    return res

##function that resizes an image based on the new height (nh) and the new width(nw).
def resize(img, nh, nw):
    res = cv.resize(img,(nw,nh), interpolation = cv.INTER_LINEAR)
    return res


##function that crops an image based on the crop width(cx) and the crop heigth(cy) values.
##the function finds
def crop(img, cx, cy):
    y, x = img.shape[:2]
    sx = x//2-(cx//2)
    sy = y//2-(cy//2)

    res = img[sy:sy+cy,sx:sx+cx]
    return(res)

def rotate(img, v):
    rows,cols = img.shape[0:2]
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),v,1)
    dst = cv.warpAffine(img,M,(cols,rows))

    return(dst)

def mirror(img):
    res = cv.flip(img, 1)

    return(res)

def grayBlur(img, scale):
    z = np.zeros(img.shape)
    cv.circle(z, (img.shape[1]//2, img.shape[0]//2), int(scale*0.9),(1,1,1),-1,8,0)
    res = cv.addWeighted(img, 4, cv.GaussianBlur(img,(0,0),scale/30),-4,128)*z+128*(1-z)
    return(res)

def scaleRadius(img,scale):
    x=img[img.shape[0]//2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r


    res = cv.resize(img,(0,0),fx=s,fy=s)
    return(res)



#for file in os.listdir(dir):
    #img = cv.imread(dir+file)
    #img = crop(img, 2400, 2400)
    #img = resize(img, 256, 256)
    #cv.imwrite('C:/Users/s161915/Documents/DataScience/Data Challenge/output/'+file+'pp.jpeg', img)


def example1(dir):
    for file in os.listdir(dir):
        img = cv.imread(dir+file)
        img = crop(img, 2200, 2200)
        img = resize(img, 256, 256)
        img = rotate(img, 270)
        img = mirror(img)
        cv.imwrite('C:/Users/s161915/Documents/DataScience/Data Challenge/example1/'+file+'pp.jpeg', img)

def example2(dir):
    for file in os.listdir(dir):
        img = cv.imread(dir+file)
        img = scaleRadius(img, 300)
        img = grayBlur(img, 300)
        cv.imwrite('C:/Users/s161915/Documents/DataScience/Data Challenge/example2/'+file+'pp.jpeg', img)

def example3(dir):
    for file in os.listdir(dir):
        img = cv.imread(dir+file)
        img = scaleRadius(img, 300)
        img = grayBlur(img, 300)
        cv.imwrite('C:/Users/s161915/Documents/DataScience/Data Challenge/example3/'+file+'pp.jpeg', img)

def preprocess(dirin, dirout):
    print('Starting preprocessing')
    start = time.time()
    i = 0
    step = 0
    x = len(os.listdir(dirin))


    for file in os.listdir(dirin):
        img = cv.imread(dirin+file)
        try:
            img = scaleRadius(img, 300)
        except:
            print(file +' gave an exception')
            pass
        img = grayBlur(img, 300)
        cv.imwrite(dirout+file+'_pp.jpeg', img)
        i += 1

        if float(i/x) > float(step/100):
            print('-- ' + str(step)+'% done, this took '+ str(time.time()-start) +' seconds so far.')
            step += 1


    print('All files succesfully saved in ' + dirout)

#preprocess(dirin, dirout)

### Blazej experiment start

def read_labels(path):
    labels = pd.read_csv(path)
    return labels

os.chdir("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1")
labels = pd.read_csv("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/trainLabels.csv")
trainImagesPath = "/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/trainImagesSample"

labels.head()
one_hot_labels = keras.utils.to_categorical(labels['level'], 5) # the labels in such format will be input to a models


dirin = "/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/trainImagesSample"
dirout = '/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/trainImageSampleEdited/'

def preprocessSick(dirin, dirout):
    rotations = [90,120,180,270]
    for file in os.listdir(dirin):
        img_org = cv.imread(dirin + "/" +file)
        img_org = resize(img_org, 400,400)
        img_org = grayBlur(img_org, 300)
        cv.imwrite(dirout+file.strip(".jpeg") + "_" + "original" + ".jpeg", img_org)
        img_mirrored = mirror(img_org)
        cv.imwrite(dirout+file.strip(".jpeg") + "_" + "mirrored" + ".jpeg", img_mirrored)
        for k in rotations:
            #img = cv.imread(path + "/" +file)
            img = rotate(img_org,k)
            #img = crop(img, 3000, 3000)
            #img = resize(img, 400,400 )
            #img = grayBlur(img,300)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + str(k) + ".jpeg", img)

        #img1 = rotate(img, 90)
        #img2 = rotate(img, 120)
        #img3 = rotate(img, 180)
        #img4 = rotate(img,270)

#preprocessSick(dirin, dirout)

labels.shape
labels['image'][0]
labels[labels['image'] == "10053_right"]['level']


image_paths = [file.strip('.jpeg') for file in os.listdir(dirin)]

validImgs = {}
for file in image_paths:
    if labels['image'].isin([file]).any():
        validImgs[file] = int(labels[labels['image'] == file]['level'])

sick_paths  = [file for file in image_paths if int(labels.loc[labels['image']==file]['level']) != 0]
sick_paths
healthy_paths  = [file for file in image_paths if int(labels.loc[labels['image']==file]['level']) == 0]
healthy_paths

def preprocessSick(dirin, dirout):
    rotations = [90,120,180,270]
    for file in sick_paths:
        try:
            img_org = cv.imread(dirin + "/" +file + ".jpeg")
            img_org = resize(img_org, 400,400)
            img_org = grayBlur(img_org, 300)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "original" + ".jpeg", img_org)
            img_mirrored = mirror(img_org)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "mirrored" + ".jpeg", img_mirrored)
            for k in rotations:
                #img = cv.imread(path + "/" +file)
                img = rotate(img_org,k)
                #img = crop(img, 3000, 3000)
                #img = resize(img, 400,400 )
                #img = grayBlur(img,300)
                cv.imwrite(dirout+file.strip(".jpeg") + "_" + str(k) + ".jpeg", img)
        except:
             print(file, "gave an exception")
             pass

preprocessSick(dirin, dirout)


def preprocess_healthy (dirin, dirout):
    for file in healthy_paths:
        try:
            img_org = cv.imread(dirin + "/" +file + ".jpeg")
            img_org = resize(img_org, 400,400)
            img_org = grayBlur(img_org, 300)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "original" + ".jpeg", img_org)
            img_mirrored = mirror(img_org)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "mirrored" + ".jpeg", img_mirrored)

        except:
             print(file, "gave an exception")
             pass


preprocess_healthy(dirin, dirout)


### Trying to create labels for new bif data set
labels.shape
labels['image'][0]
labels[labels['image'] == "10053_right"]['level']


image_paths = [file.strip('.jpeg') for file in os.listdir(dirin)]

validImgs = {}
for file in image_paths:
    if labels['image'].isin([file]).any():
        validImgs[file] = int(labels[labels['image'] == file]['level'])


paths_entire_dataset = [file.strip(".jpeg") for file in os.listdir(dirout)]
paths_entire_dataset[0].split("_",2)
paths_entire_dataset


def strip_paths(dir):
    return [file.strip(".jpeg") for file in os.listdir(dir)]

paths_entire_dataset = strip_paths(dirout)
paths_entire_dataset

def hyphen_split(a):
    if a.count("_") == 1:
        return a.split("_")[0]
    else:
        return "_".join(a.split("_", 2)[:2])

hyphen_split(paths_entire_dataset[22])

labels['image'].isin([hyphen_split(paths_entire_dataset[22])]).any()


def create_labels(paths_entire_dataset, labels):
    valid_imgs = {}
    not_valid_imgs = {}
    for file in paths_entire_dataset:
        if labels['image'].isin([hyphen_split(file)]).any():
            valid_imgs[file] = int( labels[labels['image'] == hyphen_split(file)]['level'])
        else:
            not_valid_imgs[file] = -311298
    return valid_imgs, not_valid_imgs


valid_imgs, not_valid_imgs = create_labels(paths_entire_dataset, labels)

labels_augmented = pd.DataFrame.from_dict(valid_imgs, orient = "index").reset_index()
labels_augmented.columns = ['image', 'level']

not_valid_imgs = pd.DataFrame.from_dict(not_valid_imgs, orient = "index").reset_index()
not_valid_imgs.columns = ['image', 'myBirthday']

os.getcwd()

labels_augmented.to_csv("labels_augmented.csv")
not_valid_imgs.to_csv("not_valid_imgs.csv")











one_image = os.listdir(trainImagesPath)[0]
one_image
image_path = trainImagesPath + "/" + one_image
image_path
img = cv.imread(image_path)
img = resize(img, 400,400)
img = grayBlur(img, 300)

cv.imwrite("/Users/blazejmanczak/Desktop/School/Year2/Q2/DataChallange1/" + one_image, img)
