import tensorflow as tf
import numpy as np 
import random
import cv2
import os

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator

size = 32

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


model = tf.keras.Sequential([
    Conv2D(6, 3, activation='relu'),
    AveragePooling2D(2),
    Conv2D(16, 3, activation='relu'),
    AveragePooling2D(2),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(1, activation='softmax')
])


model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



model.fit(traindata, trainlabels,
          batch_size=10,
          epochs=50,
          verbose=1,
          validation_data=(testdata, testlabels))