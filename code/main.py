import tensorflow as tf
import numpy as np 
import random
import cv2
import os
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

def load_images_from_folder(folder):
    
    images = []
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        
        if img is not None:
            
            #convert to grayscale and preprocess
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28)) 
            img = img / 255
            img = np.expand_dims(img, axis=0)
            images.append(img)
            
    return images

traingirls = load_images_from_folder("../traindata/girls")
trainguys = load_images_from_folder("../traindata/guys")
testgirls = load_images_from_folder("../testdata/girls")
testguys = load_images_from_folder("../testdata/guys")
print(np.asarray(testguys).shape)

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
    
    Conv2D(28, kernel_size=(3, 3), activation='relu',padding='same'),
    (MaxPool2D(pool_size=(2, 2), padding='same')),
    Conv2D(28, kernel_size=(3, 3), activation='relu',padding='same'),
    BatchNormalization(),
    (MaxPool2D(pool_size=(2, 2), padding='same')),
    Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(traindata, trainlabels,
          batch_size=10,
          epochs=200,
          verbose=1,
          validation_data=(testdata, testlabels))
