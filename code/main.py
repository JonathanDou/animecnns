import tensorflow as tf
import numpy as np 
import random
import cv2
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

size = 128


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

trainop = load_images_from_folder("../traindata/OP")
testop = load_images_from_folder("../testdata/OP")
trainpsycho = load_images_from_folder("../traindata/Psycho-Pass")
testpsycho = load_images_from_folder("../testdata/Psycho-Pass")
trainfma = load_images_from_folder("../traindata/FMA")
testfma = load_images_from_folder("../testdata/FMA")
traindbz = load_images_from_folder("../traindata/DBZ")
testdbz = load_images_from_folder("../testdata/DBZ")
trainerased = load_images_from_folder("../traindata/Erased")
testerased = load_images_from_folder("../testdata/Erased")
trainrailgun = load_images_from_folder("../traindata/Railgun")
testrailgun = load_images_from_folder("../testdata/Railgun")
trainzero = load_images_from_folder("../traindata/Zero")
testzero = load_images_from_folder("../testdata/Zero")
trainopm = load_images_from_folder("../traindata/OPM")
testopm = load_images_from_folder("../testdata/OPM")
trainab = load_images_from_folder("../traindata/AB")
trainfate = load_images_from_folder("../traindata/Fate")
testab = load_images_from_folder("../testdata/AB")
testfate = load_images_from_folder("../testdata/Fate")

traindata = trainab + trainfate + trainopm + trainzero + trainrailgun + trainerased + traindbz + trainfma + trainpsycho + trainop
testdata = testab + testfate + testopm + testzero + testrailgun + testerased + testdbz + testfma + testpsycho + testop

traindata = np.array(traindata)
testdata = np.array(testdata)

trainlabels = np.append(np.zeros(len(trainab)), np.ones((len(trainfate))))
testlabels = np.append(np.zeros(len(testab)), np.ones((len(testfate))))
trainlabels = np.append(trainlabels, np.ones(len(trainopm))*2)
testlabels = np.append(testlabels, np.ones(len(testopm))*2)
trainlabels = np.append(trainlabels, np.ones(len(trainzero))*3)
testlabels = np.append(testlabels, np.ones(len(testzero))*3)
trainlabels = np.append(trainlabels, np.ones(len(trainrailgun))*4)
testlabels = np.append(testlabels, np.ones(len(testrailgun))*4)
trainlabels = np.append(trainlabels, np.ones(len(trainerased))*5)
testlabels = np.append(testlabels, np.ones(len(testerased))*5)
trainlabels = np.append(trainlabels, np.ones(len(traindbz))*6)
testlabels = np.append(testlabels, np.ones(len(testdbz))*6)
trainlabels = np.append(trainlabels, np.ones(len(trainfma))*7)
testlabels = np.append(testlabels, np.ones(len(testfma))*7)
trainlabels = np.append(trainlabels, np.ones(len(trainpsycho))*8)
testlabels = np.append(testlabels, np.ones(len(testpsycho))*8)
trainlabels = np.append(trainlabels, np.ones(len(trainop))*9)
testlabels = np.append(testlabels, np.ones(len(testop))*9)

indices = np.arange(len(traindata))

traindata = traindata[indices]
trainlabels = trainlabels[indices]

indices = np.arange(len(testdata))

testdata = testdata[indices]
testlabels = testlabels[indices]

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

datagen.fit(traindata)

model = tf.keras.Sequential()

model = tf.keras.Sequential([
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(2, strides=2),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2, strides=2),
    Conv2D(128, 3, activation='relu'),
    MaxPool2D(2, strides=2),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              

model.fit(datagen.flow(traindata, trainlabels, batch_size=5),
          steps_per_epoch=int(len(traindata) / 5),
          epochs=200,
          verbose=1,
          validation_data=(testdata, testlabels))