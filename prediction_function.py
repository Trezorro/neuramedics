# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:34:27 2018

@author: 20175876
"""

import keras, tensorflow
import keras.preprocessing.image as image
import numpy as np

import tempfile, os

# defModel = keras.models.load_model('retinopathy.hdf5')
# defModel._make_predict_function()

def prediction(file, modelbytes):
    """Creates a prediction for one file of one eye"""

    if modelbytes is None:
        print("No model received for prediction, defaulting to bad test model.")
        modelpath = 'retinopathy.hdf5'
        model = keras.models.load_model(modelpath)
    else:
        # with open('tempmodel.hdf5','w+b') as modelfile:
        #     modelfile.write(modelbytes)
        #     print(' ============ modelfile:', str(modelfile.read())[:200])
        """Code gets weird here, but it was necessary."""
        tf, modelpath = tempfile.mkstemp()
        print('Made temporary file at : ', modelpath)
        try:
            os.write(tf, modelbytes)
            os.close(tf)
            model = keras.models.load_model(modelpath)
        finally:
            os.remove(modelpath)

    # outputlist = []
    #resize image to the input format
    img = image.load_img(file, target_size=(128,128,3)) ##if len(file) < 300 else image.resize()
    img = image.img_to_array(img)
    res_img = np.resize(img, (1,128,128,3))

    # graph = tensorflow.get_default_graph()
    #predict image
    # with graph.as_default():
        # labels = self.model.predict(data)
    model._make_predict_function()
    predictions = model.predict(res_img).tolist()[0]

    print('prediction: ',predictions)
    #multiclass
    # predictlist = predict.tolist()[0]
    # outputlist.append(predictlist)
    keras.backend.clear_session()
    #binary
    sumstages = predictions[1] + predictions[2] + predictions[3] + predictions[4]
    binaryOutput = [predictions[0], sumstages]
    # outputlist.append(binarylist)
    output = {
            'multi': predictions,
            'binary': binaryOutput
            }
    print("Outputting ",output)
    return output


# def get_metrics(model=model):
# """Maybe we can display model metrics like accuracy too"""
#     keras.metrics.binary_accuracy(y_true, y_pred)

if __name__ == '__main__':
    file = r'10_left.jpeg'
    print(prediction(file))
