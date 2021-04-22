import tensorflow as tf
import numpy as np 
import random
import cv2
import os

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator

size = 64

def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))

        if img is not None:

            #preprocess

            img = cv2.resize(img, (size, size)) 
            img = img / 255
            images.append(img)

    return images

traingirls = load_images_from_folder("../traindata/girls")
trainguys = load_images_from_folder("../traindata/guys")
testgirls = load_images_from_folder("../testdata/girls")
testguys = load_images_from_folder("../testdata/guys")

traindata = traingirls + trainguys  
testdata = testgirls + testguys 

traindata = np.array(traindata)
testdata = np.array(testdata)

trainlabels = np.append(np.zeros(len(traingirls)), np.ones((len(trainguys))))
testlabels = np.append(np.zeros(len(testgirls)), np.ones((len(testguys))))

indices = np.arange(len(traindata))

traindata = traindata[indices]
trainlabels = trainlabels[indices]

indices = np.arange(len(testdata))

testdata = testdata[indices]
testlabels = testlabels[indices]

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest")

datagen.fit(traindata)


model = tf.keras.Sequential([
    Conv2D(32, 3, 1, activation='relu'),
    MaxPool2D(2),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# model.fit(datagen.flow(traindata, trainlabels, batch_size=10),
#           steps_per_epoch=int(len(traindata) / 10),
#           epochs=25,
#           verbose=1,
#           validation_data=(testdata, testlabels))

model.fit(traindata, trainlabels,
          batch_size=10,
          epochs=25,
          verbose=1,
          validation_data=(testdata, testlabels))