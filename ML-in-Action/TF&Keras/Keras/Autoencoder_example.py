# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:32:18 2018

@author: jz
"""
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

(X_train, _), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255. - 0.5
X_test = X_test.astype('float32') / 255. - 0.5
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# in order to plot in a 2D figure
encoding_dim = 2

#  input placeholder
input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

'''
对自编码进行autoencoder训练，
训练完之后使用autoencoder前半部分(编码器)encoder进行预测
'''

# construct the autoencoder model,conclude encoded and decoder layers
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting，only have encoder layers
encoder = Model(input=input_img, output=encoded_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True)

# plotting
encoded_imgs = encoder.predict(X_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()



















