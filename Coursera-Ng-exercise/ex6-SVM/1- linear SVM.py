# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:56:02 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('./data/ex6data1.mat')
#print(mat.keys())
data = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
data['y'] = mat.get('y')

#数据可视化
#fig,ax = plt.subplots(figsize=(8,6))
#ax.scatter(data['X1'],data['X2'],s=50,c=data['y'],cmap='Reds')
#ax.set_title('Raw data')
#ax.set_xlabel('X1')
#ax.set_ylabel('X2')
#plt.show()

svc1 = sklearn.svm.LinearSVC(C=1,loss='hinge')
svc1.fit(data[['X1','X2']],data['y'])
svc1.score(data[['X1','X2']],data['y'])
data['SVM1 Confidence'] = svc1.decision_function(data[['X1','X2']])
fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'],data['X2'],s=50,c=data['SVM1 Confidence'],cmap='RdBu')
ax.set_title('SVM C=1')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()