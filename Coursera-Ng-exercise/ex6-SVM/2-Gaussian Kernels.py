# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:23:52 2018
高斯核函数
@author: jz
"""

import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

def gaussian_kernel(x1,x2,sigma):
    return np.exp(-np.power(x1-x2,2).sum() / (2 * (sigma ** 2)))

mat = sio.loadmat('./data/ex6data2.mat')
data = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
data['y'] = mat.get('y')

sns.set(context="notebook",style="white",palette=sns.diverging_palette(240,10,n=2))
sns.lmplot('X1','X2',hue='y',data=data,
           size=5,
           fit_reg=False,
           scatter_kws={"s":10}
           )
plt.show()

#参数kernel='rbf',rbf即表示使用的是高斯核函数，gamma = 1 / (2 * (sigma ** 2))
svc = sklearn.svm.SVC(C=100,kernel='rbf',gamma=10,probability=True)
svc.fit(data[['X1','X2']],data['y'])
#svc.score(data[['X1','X2']],data['y'])

predict_proba = svc.predict_proba(data[['X1','X2']])[:,0]
fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'],data['X2'],s=30,c=predict_proba,cmap='Reds')
plt.show()
