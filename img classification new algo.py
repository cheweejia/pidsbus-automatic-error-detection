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


def create_cae():
    # Define encoder
    conv_encoder = keras.models.Sequential([
        keras.layers.Conv2D(256, kernel_size=3, padding="SAME", activation="relu", input_shape=[150, 150, 3]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=2),

    ])

    # Define decoder
    conv_decoder = keras.models.Sequential([
        keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="SAME", activation="relu",
                                     input_shape=[37, 37, 64]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="SAME", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="SAME", activation="sigmoid"),
    ])

    # Define AE
    conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

    # Display the model's architecture
    conv_encoder.summary()
    conv_decoder.summary()

    # Compile the model
    conv_ae.compile(loss="mse", optimizer= 'adam',
                    metrics= ['accuracy'])

    return conv_ae

model = create_cae()
history = model.fit(train_images, train_labels, batch_size=250, epochs=4, validation_split = 0.2)
test_loss = model.evaluate(test_images, test_labels)