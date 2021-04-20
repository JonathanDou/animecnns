import tensorflow as tf
import numpy as np 
import random
import cv2
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
import os

resizeshape = 28

def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))

        if img is not None:

            #convert to grayscale and preprocess

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (resizeshape, resizeshape)) 
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
# print(np.asarray(testfate).shape)

traindata = trainab + trainfate + trainopm + trainzero + trainrailgun + trainerased + traindbz + trainfma + trainpsycho + trainop
testdata = testab+ testfate + testopm + testzero + testrailgun + testerased + testdbz + testfma + testpsycho + testop

traindata = np.array(traindata)
testdata = np.array(testdata)

trainlabels = np.append(np.zeros(len(trainab)), np.ones((len(trainfate))))
testlabels = np.append(np.zeros(len(testab)), np.ones((len(testfate))))
trainlabels = np.append(trainlabels, np.zeros(len(trainopm))*2)
testlabels = np.append(testlabels, np.zeros(len(testopm))*2)
trainlabels = np.append(trainlabels, np.zeros(len(trainzero))*3)
testlabels = np.append(testlabels, np.zeros(len(testzero))*3)
trainlabels = np.append(trainlabels, np.zeros(len(trainrailgun))*4)
testlabels = np.append(testlabels, np.zeros(len(testrailgun))*4)
trainlabels = np.append(trainlabels, np.zeros(len(trainerased))*5)
testlabels = np.append(testlabels, np.zeros(len(testerased))*5)
trainlabels = np.append(trainlabels, np.zeros(len(traindbz))*6)
testlabels = np.append(testlabels, np.zeros(len(testdbz))*6)
trainlabels = np.append(trainlabels, np.zeros(len(trainfma))*7)
testlabels = np.append(testlabels, np.zeros(len(testfma))*7)
trainlabels = np.append(trainlabels, np.zeros(len(trainpsycho))*8)
testlabels = np.append(testlabels, np.zeros(len(testpsycho))*8)
trainlabels = np.append(trainlabels, np.zeros(len(trainop))*9)
testlabels = np.append(testlabels, np.zeros(len(testop))*9)

indices = np.arange(len(traindata))

traindata = traindata[indices]
trainlabels = trainlabels[indices]

indices = np.arange(len(testdata))

testdata = testdata[indices]
testlabels = testlabels[indices]
print(testdata.shape)
print(testlabels.shape)
print(traindata.shape)
print(trainlabels.shape)

model = tf.keras.Sequential([
    Conv2D(10, kernel_size=(3, 3), activation='relu'),
    (MaxPool2D(pool_size=(2, 2),padding='same')),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])


model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(traindata, trainlabels,
          batch_size=5,
          epochs=20,
          verbose=1,
          validation_data=(testdata, testlabels))