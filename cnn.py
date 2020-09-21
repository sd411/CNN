# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:46:27 2020

@author: dell
"""
import keras
import tensorflow

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator




#initialising CNN

classifier = Sequential();

#1.convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = "relu"))

#2.Pooling
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

#adding second convolutional layer
classifier.add(Convolution2D(32,3,3,activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

classifier.add(Convolution2D(64,3,3,activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))


#3.Flattening
classifier.add(Flatten())

#4.Full Connection
classifier.add(Dense(output_dim = 128,activation="relu"))
classifier.add(Dense(output_dim = 128,activation="relu"))
classifier.add(Dense(output_dim = 1,activation="sigmoid"))

#5.Compiling CNN
classifier.compile(optimizer='adam' ,loss = 'binary_crossentropy', metrics=['accuracy'])


#Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch =8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)


#making new preds
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
















