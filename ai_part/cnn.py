import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


#Library for CNN Model
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class_names = ['house', 'marriage', 'tshirt', 'cop', 'firefighter', 'pain']

train_path = 'train'
test_path = 'test'
validate_path = 'validate'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=class_names, batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=class_names, batch_size=10)

validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=validate_path, target_size=(224,224), classes=class_names, batch_size=10, shuffle=False)

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
#print(labels)

#Defining the Convolutional Neural Network
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), input_shape = (224,224,3), activation='relu', padding='same'),
    MaxPooling2D(pool_size = (2, 2), strides=2),
    Conv2D(64, kernel_size=(3, 3), input_shape = (28,28,1), activation='relu', padding='same'),
    MaxPooling2D(pool_size = (2, 2), strides=2),
    Flatten(),
    Dense(units = 6, activation = 'softmax')
])

model.summary()

#Compiling
model.compile(loss ='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001) ,metrics =['accuracy'])


#Training the CNN model
history = model.fit(x=train_batches, validation_data=validation_batches, epochs = 10, verbose = 2)

model.save('saved_model/my_model_v1')

predicted_classes = model.predict(test_batches, verbose=0)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predicted_classes, axis=-1))

def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('normalized confusion matrix')
    else:
        print('confusion matrix without normalization')
    print(cm)

    tresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment='center',
            color='white' if cm[i,j] > tresh else 'black')

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('Predicted label')
    plt.show()

cm_plot_labels = class_names
plot_confusion_matrix(cm, cm_plot_labels)
