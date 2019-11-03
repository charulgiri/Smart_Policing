#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function
   
import math   # for mathematical operations
# import matplotlib.pyplot as plt    # for plotting the images

import h5py
import os
from os import listdir
from os.path import isfile, join

import cv2   # for capturing videos
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
from sklearn.model_selection import KFold
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential,Input,Model,model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, BatchNormalization, Input, InputLayer
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from keras.optimizers import SGD
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

print(tf.__version__)


# In[ ]:

h=246
datasets=['full_data_movies_kfold.pickle','full_data_peliculas.pickle', 'full_data.pickle']
pickle_file = datasets[0]  #choose the dataset to train the model

#--------------------------------make labels for the dataset-----------------------------
if pickle_file=='full_data_movies_kfold.pickle':
    h=246
    pickle_file = 'labels_movies_kfold.pickle'         #for movies dataset
    with open(pickle_file, 'rb') as f:                 #for movies dataset
    save = pickle.load(f)                              #for movies dataset
    Y= np.array(save)                                  #for movies dataset
elif pickle_file=='full_data_peliculas.pickle':   
    h=201
    Y=list([1]*100)+list([0]*101)                     #for peliculas dataset
    Y=np.array(Y)                                     #for peliculas dataset

else:
   h=1000                                             #for hockey dataset
   label1 = list([1]*500)
   label2=list([0]*500)
   Y=label1+label2
   Y=np.array(Y)

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X = np.array(save)


# In[ ]:

                      
Y=np_utils.to_categorical(Y)
import sys
X.nbytes


# In[ ]:


normalized_X=np.zeros((h,10,3,60,90))

X=np.reshape(X,(h,10,3,60,90))
x=X                       #for movies dataset
#x,y=shuffle(X,Y)         #for peliculas dataset



scaler=StandardScaler()
for i in range(h):
    m=0
    for j in range(10):
        for k in range(3):
            
            x_2d=x[i][m][k]
#             print(x_2d.shape)
            scaler.fit(x_2d)
            normalized_X[i][j][k] = scaler.transform(x_2d)

   
            

normalized_X.nbytes

normalized_X=np.reshape(normalized_X,(h,10,60,90,3))

#x,y=shuffle(normalized_X,y)     #for peliculas dataset
x=normalized_X          #for movies dataset
y=Y
#x,y=shuffle(x,y)       #for peliculas dataset

#x_eval=x[800:]
#y_eval=y[800:]
#x=x[:800]
#y=y[:800]
# In[ ]:


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        global Accuracy
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        if acc>Accuracy:
            model.save('best_model_search2.h5')
            Accuracy=acc
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
# In[ ]:


def step_decay(epoch):
    global learning
    initial_lrate = learning
   # drop =0.1  #best performance 98%
   # epochs_drop = 400.0 #best performance
    drop =0.1  #tune for best performance 98%
    epochs_drop = 400.0 # tune for best performance
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[ ]:



#batch_size = 10
batch_size=20
epochs =1000 #for 98% best performance 1000 epoch
num_classes = 2

tensorboard = TensorBoard(batch_size=batch_size)
def Conv(filters=16, kernel_size=(3,3,3), activation='relu', input_shape=None,kernel_regularizer=None):
    if input_shape:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, input_shape=input_shape)
    else:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, kernel_regularizer=kernel_regularizer)

# Define Model
def CNN(input_dim, num_classes):
    model = Sequential()

    model.add(Conv(32, (7,7,7), input_shape=input_dim))
    model.add(Conv(32,(5,1,1)))
    model.add(Conv(32,(1,5,1)))
    model.add(Conv(32,(1,1,5))) 
#    model.add(Conv(32, (5,5,5)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
#    model.add(Conv(48, (5,5,5)))

#     model.add(Dropout(0.25))

#     model.add(Conv(32, (3,3,3)))
    model.add(Conv(64,(3,1,1)))
    model.add(Conv(64,(1,3,1)))
    model.add(Conv(64,(1,1,3))) 
#    model.add(Conv(64, (3,3,3)))
#     model.add(Conv(128, (3,3,3)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(BatchNormalization())
    model.add(MaxPool3D())
#     model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(512,activation='relu'))

    model.add(Dense(128,activation='linear', kernel_regularizer=regularizers.l2(0.2))) #,kernel_regularizer=regularizers.l2(0.01)
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax'))

    return model

# Train Model
def train(optimizer, scheduler):
    global model
    global history
    global Accuracy
    print("Training...", learning)
#    model.load_weights('Benchmark_movies_kfold2.h5')
#    model2=tf.keras.models.load_model('Benchmark_movie_k_fold.h5')
#    print(model2.optimizer.lr)
#    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    lrate = LearningRateScheduler(step_decay)
    print(model.summary())

  #  mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max',verbose=1,period=1, save_best_only=True)
    kfold=KFold(5,False,None)      #for movies dataset set value to false else true
    results = []
    count=0
    start=0
    end=50
    for train,test  in kfold.split(x):
        model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
        count+=1
        X_train, X_test, Y_train, Y_test = x[train], x[test], y[train], y[test]      #for peliculas and movies dataset
        X_train,Y_train=shuffle(X_train,Y_train)         #for movies dataset
        X_test,Y_test=shuffle(X_test,Y_test)             #for movies dataset
        history=model.fit(X_train,Y_train, validation_data=(X_test,Y_test),batch_size=batch_size, epochs=epochs,
                    verbose=1,callbacks=[lrate])
        eval=model.evaluate(X_test,Y_test)
        print(count,eval[0],eval[1])
        results.append(eval[1])
 #model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#        if eval[1]<0.90:
#            break
    Avg_perform = (sum(results)/5)*100
    Accuracy=Avg_perform
    if Avg_perform>=98.0:
        model.save("Benchmark_movies_kfold_correct_500_epoches.h5")
    print('Average accuracy:', round(Avg_perform, 2))
#    eval=model.evaluate(x_eval,y_eval)
#   print(eval[0],eval[1])

def evaluate():
    global model

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    print(accuracy_score(pred,y_test))
    # Heat Map
    array = confusion_matrix(y_test, pred)
    cm = pd.DataFrame(array, index = range(10), columns = range(10))
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True)
    plt.show()

def save_model():
    global model

    model_json = model.to_json()
    with open('10_model_3D.json', 'w') as f:
        f.write(model_json)

    model.save_weights('10_model_3D.h5')

    print('Model Saved.')

def load_model():
    f = open('model_3D.json', 'r')
    model_json = f.read()
    f.close()

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model/model_3D.h5')

    print("Model Loaded.")
    return loaded_model

if __name__ == '__main__':

#     optimizer = RMSprop(lr=0.1, rho=0.5, epsilon=1e-08, decay=0.0)
#     optimizer=SGD(lr=0.00001, momentum=0.95)
    Accuracy=0.0
    learning=0.000048
#    optimizer = Nadam(lr=0.00011)    #for movies dataset
    optimizer = Nadam(lr=learning)
#    optimizer=Nadam(lr=0.00001)      #for peliculas dataset
    scheduler = ReduceLROnPlateau(monitor='val_acc',  verbose=1)

#    model = CNN((10,60,90,3), 2)
  #  while Accuracy<90:
    optimizer = Nadam(lr=learning)
    model = CNN((10,60,90,3), 2)
    train(optimizer, scheduler)
 #   learning=np.random.uniform(0.00004,0.00005)
#    learning=round(learning,6)

