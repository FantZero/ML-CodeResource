# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:38:06 2018

@author: jz
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import train_test_split, standardize, to_categorical, normalize
from utils import mean_squared_error, accuracy_score
from xgboost import XGBoost

def main():
    print ('-- XGBOOST --')
    
    #load temperature data
    data = pd.read_csv('TempLinkoping2016.txt', sep='\t')
    
    time = np.atleast_2d(data['time'].as_matrix()).T
    temp = np.atleast_2d(data['temp'].as_matrix()).T
    
    X = time.reshape((-1, 1))
    X = np.insert(X, 0, values=1, axis=1)
    y = temp[:, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    model = XGBoost()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
#    y_pred_line = model.predict(X)
    
    print (y_pred[0:5])
    
    cmap = plt.get_cmap('viridis')
    
    mse = mean_squared_error(y_test, y_pred)
    
    print ('Mean Squared Error:', mse)
    
    m1 = plt.scatter(336 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(336 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(336 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle('Regression Tree')
    plt.title('MSE: %.2f' % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1,m2,m3), ('Traning data','Test data','Prediction'), loc='lower right')
    plt.show()
    
main()
    






















