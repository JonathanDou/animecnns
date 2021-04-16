import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from PIL import Image
import glob
from skimage.transform import resize
from keras.layers import Dropout
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            image = resize(img, (16, 16), 3)
            images.append(img)
    return images

folder="traindata/girls"
traingirls = load_images_from_folder(folder)
folder="traindata/guys"
trainguys = load_images_from_folder(folder)
folder="testdata/girls"
testgirls = load_images_from_folder(folder)
folder="testdata/guys"
testguys = load_images_from_folder(folder)

(X_train, y_train), (X_test, y_test) = (traingirls, trainguys), (testgirls, testguys)

model = Sequential()

model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=128,
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test))
