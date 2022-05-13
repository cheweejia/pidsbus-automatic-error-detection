import numpy as np
import pandas as pd
import os
import seaborn as sn; sn.set(font_scale=1.4)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_names:i for i, class_names in enumerate(class_names)}
nb_classes = len(class_names)
print(class_names_label)

IMAGE_SIZE = (150,150)


def load_data():
    DIRECTORY = 'D:\intel image'
    CATEGORY = ['seg_train', 'seg_test']

    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        images = []
        labels = []
        print(path)
        for folder in os.listdir(path):    #Iterate through each class folder
            label = class_names_label[folder]
            for file in os.listdir(os.path.join(path, folder)):  #iterate each photo per class
                img_path = os.path.join(os.path.join(path, folder), file)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label)
        images = np.array(images, dtype= 'float32')
        labels = np.array(labels, dtype= 'int32')
        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

model = keras.Sequential([layers.Conv2D(32, (3,3), activation = 'relu', input_shape= (150, 150, 3)),
layers.MaxPooling2D(2,2), layers.Conv2D(32, (3,3), activation = 'relu'),
layers.MaxPooling2D(2,2), layers.Flatten(), layers.Dense(128, activation=tf.nn.relu),
layers.Dense(6, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=120, epochs=3, validation_split = 0.2)
test_loss = model.evaluate(test_images, test_labels)
result = model.

