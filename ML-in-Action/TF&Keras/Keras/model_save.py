# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:23:00 2018

@author: jz
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  #随机化X的数据
y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

X_train, y_train = X[:160], y[:160]
X_test, y_test = X[160:], y[160:]

model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

## save，保存前测试前两个例子
#print ('test before save: ', model.predict(X_test[0:2]))
#model.save('./save_files/model_save.h5')
#del model
## load
#model = load_model('./save_files/model_save.h5')
#print('test after load: ', model.predict(X_test[0:2]))

# save and load weights
#model.save_weights('./save_files/my_model_weights.h5')
#model.load_weights('./save_files/my_model_weights.h5')

# save and load fresh network without trained weights
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)















