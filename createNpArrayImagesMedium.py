from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
import os
import keras

os.chdir("/mnt/server-home/TUE/20184102/")

def completeWithAugmentedData(X_tr,Y_tr,limit):
  toAdd=limit-len(X_tr)
  if (len(X_tr)<limit):
    datagen = ImageDataGenerator(
    #  width_shift_range=0.1,
      #height_shift_range=0.1,
      #rotation_range=15,
      zoom_range=0.08,
      horizontal_flip=True)
    gentrain = datagen.flow(np.asarray(X_tr), batch_size=1)
    igen=0
    for gt in gentrain:
      X_tr.append(gt[0,:,:,:])
      Y_tr.append(Y_tr[0])
      igen += 1
      if (igen >= toAdd):
        break
  #print len(X_tr),len(Y_tr)
  return [X_tr,Y_tr]

def read_data_small(labels_dim):
    input_dir="/mnt/server-home/TUE/20184102/datasets/train"
    labels= pd.read_csv("/mnt/server-home/TUE/20184102/datasets/trainLabels.csv")
    labels = labels.sample(frac=1, random_state=1998).reset_index()[:28000]
    print("Shape of labels in triaining is:", labels.shape)
    print("#examples of healthy: ", sum(labels['level'] == 0)
    print("#Examples of class 1 and 2:",sum((labels['level'] == 1) | (labels['level'] == 2)))
    print("#examples of class 3 and 4:", sum((labels['level'] == 3) | (labels['level'] == 4)))
    X_train0 = []
    X_train1 = []
    X_train2 = []
      #X_train3 = []
      #X_train4 = []

    Y_train0 = []
    Y_train1 = []
    Y_train2 = []
      #Y_train3 = []
      #Y_train4 = []

    limit=2500

    for idx,filename in enumerate(labels["image"]):
        #print("Idx:", idx, "filename", filename)
        #print("labels[level][idx]", labels["level"][idx])
        #img = load_img(input_dir+"/"+str(labels["level"][idx])+"/"+ filename+".tiff", target_size = (256, 256)) # this is a PIL image
        img = load_img(input_dir + "/" + filename + ".jpeg", target_size = (256, 256))
        x = img_to_array(img) # this is a Numpy array with shape (256, 256, 3)

        if ( (labels["level"][idx]==0) and (len(X_train0)<limit) ):
            X_train0.append(x)
            #Y_train0.append(labels["level"][idx])
            Y_train0.append(0)


        if ( ((labels["level"][idx]==1) or (labels["level"][idx]==2)) and (len(X_train1)<limit)):
            X_train1.append(x)
            #Y_train1.append(labels["level"][idx])
            Y_train1.append(1)


        if ( ((labels["level"][idx]==3) or (labels["level"][idx]==4)) and (len(X_train2)<limit)):
            X_train2.append(x)
            #Y_train2.append(labels["level"][idx])
            Y_train2.append(2)


        """
        if ((labels["level"][idx]==3)and(len(X_train3)<limit)):
          X_train3.append(x)
          Y_train3.append(labels["level"][idx])

        if ((labels["level"][idx]==4)and(len(X_train4)<limit)):
          X_train4.append(x)
          Y_train4.append(labels["level"][idx])
        """
    [X_train0,Y_train0]= completeWithAugmentedData(X_train0,Y_train0,limit)
    [X_train1, Y_train1] = completeWithAugmentedData(X_train1, Y_train1,limit)
    [X_train2, Y_train2] = completeWithAugmentedData(X_train2, Y_train2,limit)
    #[X_train3, Y_train3] = completeWithAugmentedData(X_train3, Y_train3,limit)
    #[X_train4, Y_train4] = completeWithAugmentedData(X_train4, Y_train4,limit)

    X_train=np.concatenate((np.asarray(X_train0),np.asarray(X_train1),np.asarray(X_train2)))#,np.asarray(X_train3),np.asarray(X_train4)),axis=0)
    Y_train=np.concatenate((np.asarray(Y_train0),np.asarray(Y_train1),np.asarray(Y_train2)))#,np.asarray(Y_train3),np.asarray(Y_train4)),axis=0)

    X_train = X_train.astype('float16') / 255.
    Y_train = keras.utils.to_categorical(Y_train, labels_dim)

    print ("TrainData size", X_train.shape, Y_train.shape)
    np.savez_compressed("/mnt/server-home/dc_group08/data/npz/trainDataMediumTrenary.npz",X_train=X_train,Y_train=Y_train)

def read_data_small_test(labels_dim):
    input_dir="/mnt/server-home/TUE/20184102/datasets/train"
    labels= pd.read_csv("/mnt/server-home/TUE/20184102/datasets/trainLabels.csv")
    labels = labels.sample(frac=1, random_state=1998)[28000:]
    labels = labels.reset_index()
    print("Shape of labels in testing is:", labels.shape)
    print("#examples of healthy: ", sum(labels['level'] == 0)
    print("#Examples of class 1 and 2:",sum((labels['level'] == 1) | (labels['level'] == 2)))
    print("#examples of class 3 and 4:", sum((labels['level'] == 3) | (labels['level'] == 4)))
    X_test0 = []
    X_test1 = []
    X_test2 = []
    #X_test3 = []
    #X_test4 = []

    Y_test0 = []
    Y_test1 = []
    Y_test2 = []
    #Y_test3 = []
    #Y_test4 = []

    limit= 600

    for idx,filename in enumerate(labels["image"]):
        #img = load_img(input_dir+"/"+str(labels["level"][idx])+"/"+ filename+".tiff", target_size = (256, 256)) # this is a PIL image
        img = load_img(input_dir + "/" + filename + ".jpeg", target_size = (256, 256))
        x = img_to_array(img) # this is a Numpy array with shape (256, 256, 3)

        if ((labels["level"][idx]==0)and(len(X_test0)<limit)):
          X_test0.append(x)
          #Y_test0.append(labels["level"][idx])
          Y_test0.append(0)

        if ( ((labels["level"][idx]==1) or (labels["level"][idx]==2)) and (len(X_test1)<limit)):
          X_test1.append(x)
          #Y_test1.append(labels["level"][idx])
          Y_test1.append(1)


        if ( ((labels["level"][idx]==3) or (labels["level"][idx]==4)) and (len(X_test2)<limit)):
          X_test2.append(x)
          #Y_test2.append(labels["level"][idx])
          Y_test2.append(2)
        """
        if ((labels["level"][idx]==3)and(len(X_test3)<limit)):
          X_test3.append(x)
          Y_test3.append(labels["level"][idx])

        if ((labels["level"][idx]==4)and(len(X_test4)<limit)):
          X_test4.append(x)
          Y_test4.append(labels["level"][idx])
        """
    [X_test0,Y_test0]=completeWithAugmentedData(X_test0,Y_test0,limit)
    [X_test1, Y_test1] = completeWithAugmentedData(X_test1, Y_test1,limit)
    [X_test2, Y_test2] = completeWithAugmentedData(X_test2, Y_test2,limit)
    #[X_test3, Y_test3] = completeWithAugmentedData(X_test3, Y_test3,limit)
    #[X_test4, Y_test4] = completeWithAugmentedData(X_test4, Y_test4,limit)

    X_test=np.concatenate((np.asarray(X_test0),np.asarray(X_test1),np.asarray(X_test2)))#,np.asarray(X_test3),np.asarray(X_test4)),axis=0)
    Y_test=np.concatenate((np.asarray(Y_test0),np.asarray(Y_test1),np.asarray(Y_test2)))#,np.asarray(Y_test3),np.asarray(Y_test4)),axis=0)

    X_test = X_test.astype('float16') / 255.
    Y_test = keras.utils.to_categorical(Y_test, labels_dim)

    print ("TrainData size", X_test.shape, Y_test.shape)
    np.savez_compressed("/mnt/server-home/dc_group08/data/npz/testDataMediumTrenary.npz",X_test=X_test,Y_test=Y_test)

read_data_small(3)
read_data_small_test(3)

"""
def read_data(labels_dim):
  input_dir="data/trainData"
  labels=pd.read_csv("data/trainData.csv")

  X_train=[]
  Y_train=[]

  for idx,filename in enumerate(labels["image"]):
    img = load_img(input_dir+"/"+str(labels["level"][idx])+"/"+ filename+".tiff") # this is a PIL image
    x = img_to_array(img) # this is a Numpy array with shape (128, 128, 3)
    X_train.append(x)
    Y_train.append(labels["level"][idx])

  X_train=np.asarray(X_train)
  Y_train=np.asarray(Y_train)
  X_train = X_train.astype('float16') / 255.
  Y_train = keras.utils.to_categorical(Y_train, labels_dim)

  print ("TrainData size", X_train.shape, Y_train.shape)
  np.savez_compressed("data/trainData.npz",X_train=X_train,Y_train=Y_train)




def read_train_data():
  data=np.load("data/trainData.npz")
  X_train=data["X_train"]
  Y_train=data["Y_train"]
  return [X_train,Y_train]

def read_test_data():
  data=np.load("data/testData.npz")
  X_test=data["X_test"]
  Y_test=data["Y_test"]
  return [X_train,Y_train]

[X_train,Y_train]=read_train_data()
print "bau"
for j in range (100000000):
  c=j+j*1.2321
[X_test,Y_test]=read_test_data()
"""
