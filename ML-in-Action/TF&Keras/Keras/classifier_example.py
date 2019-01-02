# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:23:00 2018

@author: jz
"""
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# X shape (60000,28x28)  y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# another way to build neural net
model = Sequential([
        Dense(32, input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
        ])

# another way to define optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# add metrics to get more results want to see
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

# another way to train the model, epoch means练过程中数据将被“轮”多少次
model.fit(X_train, y_train, epochs=2, batch_size=32)

# evaluate the model with the metrics we define earlier
loss, accuracy = model.evaluate(X_test, y_test)

print ('test loss:', loss)
print ('test accuracy:', accuracy)


















