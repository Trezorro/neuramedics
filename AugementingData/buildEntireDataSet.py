import numpy as np
import cv2 as cv
import os
import time
import pandas as pd

dirin = '/mnt/server-home/TUE/20184102/datasets/train/'
dirout = '/mnt/server-home/dc_group08/data/preprocessingBlazej/train_images_FullRes/'
labels = pd.read_csv('/mnt/server-home/TUE/20184102/datasets/trainLabels.csv')

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

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return (img_cropped)

image_paths = [file.strip('.jpeg') for file in os.listdir(dirin)]
sick_paths  = [file for file in image_paths if int(labels.loc[labels['image']==file]['level']) != 0]
healthy_paths = [file for file in image_paths if int(labels.loc[labels['image']==file]['level']) == 0]

def preprocess_sick(dirin, dirout):

    print('Starting preprocessing of sick people')
    start = time.time()
    i = 0
    step = 0
    x = len(sick_paths)

    rotations = [90,120,210,270]

    for file in sick_paths:
        try:
            img_org = cv.imread(dirin + "/" +file + '.jpeg')
            img_org = scaleRadius(img_org, 300)
            img_org = grayBlur(img_org, 300)
            img_org = crop_img(img_org, scale=0.8)
            #img_org = resize(img_org, 256,256)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "original" + ".jpeg", img_org)
            img_mirrored = mirror(img_org)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "mirrored" + ".jpeg", img_mirrored)
            for k in rotations:
                img = rotate(img_org,k)
                img = crop_img(img, scale=0.88)
                cv.imwrite(dirout+file.strip(".jpeg") + "_" + str(k) + ".jpeg", img)
            i+=1
        except:
            print(file +' gave an exception')
            pass

        if float(i/x) > float(step/100):
            print('-- ' + str(step)+'% done, this took '+ str(time.time()-start) +' seconds so far.')
            step += 1
    print('All sick files succesfully saved in ' + dirout)

preprocess_sick(dirin, dirout)

def preprocess_healthy(dirin, dirout):

    print('Starting preprocessing of healthy people')
    start = time.time()
    i = 0
    step = 0
    x = len(healthy_paths)

    for file in healthy_paths:
        try:
            img_org = cv.imread(dirin + "/" +file + '.jpeg')
            img_org = scaleRadius(img_org, 300)
            img_org = grayBlur(img_org, 300)
            img_org = crop_img(img_org, scale=0.8)
            #img_org = resize(img_org, 256,256)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "original" + ".jpeg", img_org)
            img_mirrored = mirror(img_org)
            cv.imwrite(dirout+file.strip(".jpeg") + "_" + "mirrored" + ".jpeg", img_mirrored)
            i+=1
        except:
            print(file +' gave an exception')
            pass

        if float(i/x) > float(step/100):
            print('-- ' + str(step)+'% done, this took '+ str(time.time()-start) +' seconds so far.')
            step += 1
    print('All healthy files succesfully saved in ' + dirout)

preprocess_healthy(dirin, dirout)
