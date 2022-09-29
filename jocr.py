# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import cv2
import sys


def CNN():
    # Function to implement a CNN model
    num_classes = 73
    
    ## CNN model
    model = Sequential()    # Initialising the CNN
    model.add(Rescaling(1./255)) # Scale data 0 to 1
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[48, 48, 3])) # Convolution
    model.add(MaxPool2D())  # Pooling layer
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu')) # Adding a second convolutional layer
    model.add(MaxPool2D())
    model.add(Flatten())    # Flattening
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))  # Full Connection
    model.add(Dense(units=num_classes, activation='softmax'))   # Output Layer
    return model


def model_train():
    ##Function to load data and train the model
    
    ## Importing dataset
    batch_size = 32
    img_height = 48
    img_width = 48
    
    ## Creating the training set
    train = image_dataset_from_directory(
      'hiragana73/hiragana73/',
      validation_split=0.2,
      subset="training",
      label_mode = 'int',
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    ## Creating the validation set
    test = image_dataset_from_directory(
      'hiragana73/hiragana73/',
      validation_split=0.2,
      subset="validation",
      label_mode = 'int',
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    classNames = train.class_names   
    
    model = CNN()
    
    # Compile
    model.compile(optimizer='adam',loss=SparseCategoricalCrossentropy(),metrics=['accuracy'])
    
    ## Train
    history = model.fit(train,validation_data=test,epochs=3)
    
    ## Model performance graphs
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    return model, classNames



if __name__ == "__main__":
    
    
    model, classNames = model_train()
    
    
    ### Enter path to the image to be tested
    path = 'hiragana73/1903_1913342_0166.png'
    
    test_image = cv2.imread(path)   #load image to be tested
    if test_image is None:
        print('Image is not loaded!') #Exit if image is not loaded properly
        sys.exit()
    
    else:
        test_image = np.expand_dims(test_image, axis = 0)   # extending to a 4th dimention to be passed through the model
                
        ### Predicting the class of the test_image
        result = model.predict(test_image)
        clss = np.argmax(result, 1)[0]
        print('Character = ', classNames[clss])
        
        marks = result[0,clss] * 10   # Probabilities are from 0 to 1. Hence, multiply by 10 to change the range 0 to 10 
        marks = np.round(marks,0)  # Round up to closest integer for simplicity
        print('Points scored = ', marks)
        


