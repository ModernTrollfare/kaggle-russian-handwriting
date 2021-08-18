# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, LeakyReLU, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import SGD, rmsprop

import pandas as pd
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

import h5py
dataset_path = '../input/LetterColorImages_123.h5'
dataset = h5py.File(dataset_path, 'r')
# Any results you write to the current directory are saved as output.
train_data = np.array(dataset['images'])
train_data = train_data.astype('float32')/255
train_class_label = np.array(dataset['labels'])-1
train_background_label = np.array(dataset['backgrounds'])

class_onehot = keras.utils.to_categorical(train_class_label, len(set(train_class_label)))
background_onehot = keras.utils.to_categorical(train_background_label, len(set(train_background_label)))

bd_x, bv_x, bd_y, bv_y = train_test_split(train_data,background_onehot,test_size=0.25, random_state = 1)
cd_x, cv_x, cd_y, cv_y = train_test_split(train_data,class_onehot,test_size=0.25, random_state = 1)

def prep_model(dense_length):
    model = Sequential()
    #model.add(Conv2D(32, (5, 5), padding='same', input_shape=train_data.shape[1:]))
    #model.add(LeakyReLU(alpha=0.02))
    
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    #model.add(Conv2D(196, (5, 5)))
    #model.add(LeakyReLU(alpha=0.02))
    
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    #model.add(GlobalMaxPooling2D())
    
    #model.add(Dense(1024))
    #model.add(LeakyReLU(alpha=0.02))
    #model.add(Dropout(0.5)) 
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                activation='relu',
                 input_shape=train_data.shape[1:]))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_length, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

class_model = prep_model(len(set(train_class_label)))
bg_model = prep_model(len(set(train_background_label)))

class_model.fit(cd_x, cd_y, epochs=64, batch_size = 128)
bg_model.fit(bd_x, bd_y, epochs=64, batch_size = 128)

class_model.evaluate(cv_x, y=cv_y)
bg_model.evaluate(bv_x, y=bv_y)

pred_b = bg_model.predict(bv_x)
pred_c = class_model.predict(cv_x)

