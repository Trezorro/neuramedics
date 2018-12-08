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
    
def pp(img):
    img = resize(img, 128, 128)
    return(img)    
    