import keras
import keras.preprocessing.image as image
import numpy as np
import os
import base64
import plotly
import plotly.graph_objs as go

import matplotlib.pyplot as plt

SHOWGRAPH = False
dataDir = 'sample/' #include last slash
model = keras.models.load_model('retinopathy.hdf5')

x =[]

for file in os.listdir("./data/" + dataDir):
    img = image.load_img('data/' + dataDir + file, target_size=(128, 128, 3))
    plt.imshow(img)
    img = image.img_to_array(img)
    x = np.resize(img, (1, 128,128,3))
    prediction = model.predict(x)
    print("Image {:<15}:    {}\n".format(file,prediction))



    if SHOWGRAPH:
        encoded_image = base64.b64encode(open("./data/sample/"+file,'rb').read())

        plotly.offline.plot({
        "data": [go.Bar(x=["Healthy","Stage 1","Stage 2","Stage 3","Stage 4"], y=prediction[0])],
        "layout": go.Layout(title="Prediction",
                            images= [dict(
                  source= 'data:image/jpg;base64,{}'.format(encoded_image.decode()),
                  xref= "0",
                  yref= "0",
                  x= 1,
                  y= 1,
                  sizex= 0,
                  sizey= 0,
                  sizing= "contain",
                  opacity= 0.8,
                  layer= "above")])
        }, auto_open=True)
        # break #uncomment if you want to just test 1 picture
    plt.show()
