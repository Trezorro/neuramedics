# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:34:27 2018

@author: 20175876
"""

import keras
import keras.preprocessing.image as image
import numpy as np

model = keras.models.load_model('retinopathy.hdf5')

def prediction(file):
    """Creates a prediction for one file of one eye"""
    outputlist = []
    #resize image to the input format
    img = image.load_img(file, target_size=(128,128,3))
    img = image.img_to_array(img)
    res_img = np.resize(img, (1,128,128,3))

    #predict image
    predict = model.predict(res_img)
    print('prediction: ',predict)
    #multiclass
    predictlist = predict.tolist()[0]
    outputlist.append(predictlist)

    #binary
    sumstages = predictlist[1] + predictlist[2] + predictlist[3] + predictlist[4]
    binarylist = [predictlist[0]]
    binarylist.append(sumstages)
    outputlist.append(binarylist)

    return outputlist


if __name__ == '__main__':
    file = r'10_left.jpeg'
    print(prediction(file))
