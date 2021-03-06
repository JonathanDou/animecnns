import tensorflow as tf
import numpy as np 
import random
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator

size = 32
oimages = [] 

def load_images_from_folder(folder, test):

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))

        if img is not None:

            #preprocess
            
            if test:
                oimages.append(img)
                
            img = cv2.resize(img, (size, size)) 
            img = img / 255
            images.append(img)

    return images

traingirls = load_images_from_folder("../traindata/girls", False)
trainguys = load_images_from_folder("../traindata/guys", False)
testgirls = load_images_from_folder("../testdata/girls", True)
testguys = load_images_from_folder("../testdata/guys", True)

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

oimages = np.array(oimages)
oimages = oimages[indices]


model = tf.keras.Sequential([
    Conv2D(6, 3, activation='relu'),
    AveragePooling2D(),
    Conv2D(16, 3, activation='relu'),
    AveragePooling2D(),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(traindata, trainlabels,
          batch_size=5,
          epochs=50,
          verbose=1,
          validation_data=(testdata, testlabels))

model.summary()

predicted = model.predict(testdata).T  

result = np.absolute(np.array(testlabels).T-predicted)


for i in range(len(result)):
    if not result[0][i] == 0:
        plt.imshow(oimages[i])
        
plt.show()