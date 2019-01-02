# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:46:02 2018

@author: jz
"""
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  #随机化X的数据
y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))    #最终 W应该接近0.5,b接近2
plt.scatter(X, y)
plt.show()

X_train, y_train = X[:160], y[:160]
X_test, y_test = X[160:], y[160:]

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

# training
for step in range(301):
    cost = model.train_on_batch(X_train, y_train)
    if step % 100 == 0:
        print ('cost:',cost)
# test
cost = model.evaluate(X_test, y_test, batch_size=40)
W, b = model.layers[0].get_weights()
print ('Weights:',W, '\nbiases=',b)

 # plotting the prediction
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()




















