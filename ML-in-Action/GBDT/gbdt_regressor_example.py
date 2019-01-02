# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:45:42 2018

@author: jz
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

from utils import train_test_split, standardize, to_categorical
from utils.loss_functions import SquareLoss
from utils import mean_squared_error,accuracy_score,Plot
from utils.misc import bar_widgets
from gbdt import GBDTRegressor

def main():
    print '-- Grandient Boosting Regression --'
    
    data = pd.read_csv('TempLinkoping2016.txt', sep='\t')
    
    time = np.atleast_2d(data['time'].as_matrix()).T
    temp = np.atleast_2d(data['temp'].as_matrix()).T
    
    X = time.reshape((-1,1))
    X = np.insert(X, 0, values=1, axis=1)
    y = temp[:, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    model = GBDTRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cmap = plt.get_cmap('viridis')
    
    mse = mean_squared_error(y_test, y_pred)
    
    print 'Mean Squared Error:',mse
    
    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()

main()










