# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:57:15 2019

@author: edumu
"""
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Importing the libraries
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback

import tensorflow as tf

from azureml.core import Run

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

#Read the arguments:
# - data folder
# - batchsize
# - x_filename
# - y_filename
# - data training size
# - num epochs
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--x_filename', type=str, dest='x_filename', help='Filename with training data')
parser.add_argument('--y_filename', type=str, dest='y_filename', help='Filename with label data')
parser.add_argument('--training_size', type=str, dest='training_size', help='Size of training dataset')
parser.add_argument('--n_epochs', type=int, dest='n_epochs', help='Number of epochs')

args = parser.parse_args()

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)
#Load the data and labels
# Files were created previously and zipped in a readable format for numpy array
# Load npz file containing image arrays
x_npz = np.load(data_folder+'/'+args.x_filename)
x = x_npz['arr_0']
# Load binary encoded labels for Lung Infiltrations: 0=Not_infiltration 1=Infiltration
y_npz = np.load(data_folder+'/'+args.y_filename)
y = y_npz['arr_0']

# Define global variables
n_epochs = args.n_epochs
batch_size = args.batch_size
num_classes = 3
input_width=128
input_height=173

# Create training, validation and test datasets
from sklearn.model_selection import train_test_split

# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_valtest, y_train, y_valtest = train_test_split(x,y, test_size=0.2, random_state=1, stratify=y)

# Second split the 20% into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1, stratify=y_valtest)

training_set_size = X_train.shape[0]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

#Reshape input data for Tensorflow format  Num,Width,Height,Channels
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2] , 1).astype('float32')
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2] , 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2] , 1).astype('float32')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

#Build the cnn model
# A simple CNN conv2d-MAxpool-Dropout-Conv2D-MAxpool-Dropout-Conv2D-Maxpool-Dropout
# Then a fully connected layer
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(input_width,input_height,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Change in dropout keep_prob por rate, as 1- keeo:prob
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Define loss function and optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])

#Train the model and save the parameters
history= model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=2,
          validation_data=(X_val, y_val),
          callbacks=[LogRunMetrics()])

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)

## log the metrics
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

run.log("Training size", args.training_size)
print('Training size:', args.training_size)

#Plot the accuracy and loss along the epochs
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(131)
ax.set_title('Acc vs Loss ({} epochs)'.format(n_epochs), fontsize=14)
ax.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
ax.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#Plot the accuracy and loss of training and validation datasets
ax=fig.add_subplot(132)
ax.plot(epochs, acc, 'blue', label='Training acc')
ax.plot(epochs, val_acc, 'red', label='Validation acc')
ax.set_title('Training and validation accuracy')
ax.legend()

ax=fig.add_subplot(133)
ax.plot(epochs, loss, 'blue', label='Training loss')
ax.plot(epochs, val_loss, 'red', label='Validation loss')
ax.set_title('Training and validation loss')
ax.legend()
                 
# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
model.save('./outputs/model/model_full.h5')
print("model saved in ./outputs/model folder")
