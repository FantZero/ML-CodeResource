# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:33:30 2018
    在本练习中，我们将使用支持向量机（SVM）来构建垃圾邮件分类器。 我们将从一些简单的2D数据集开始使用SVM来查看它们的
工作原理。然后，我们将对一组原始电子邮件进行一些预处理工作，并使用SVM在处理的电子邮件上构建分类器，以确定它们是否为
垃圾邮件。4-spam....py中已经写了解决代码
@author: jz
"""
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
import matplotlib.pyplot as plt

def gaussian_kernel(x1,x2,sigma):
    return np.exp(-(np.sum((x1-x2) ** 2) / (2 * (sigma ** 2))))

raw_data = loadmat("./data/ex6data2.mat")
data = pd.DataFrame(raw_data['X'],columns=['X1','X2'])
data['y'] = raw_data['y']

#positive = data[data['y'].isin([1])]
#negative = data[data['y'].isin([0])]
#
#fig,ax = plt.subplots(figsize=(12,8))
#ax.scatter(positive['X1'],positive['X2'],s=50,marker='x',label='positive')
#ax.scatter(negative['X1'],negative['X2'],s=50,marker='o',label='negative')
#ax.legend()
#plt.show()

svc = svm.SVC(C=100,gamma=10,probability=True)
svc.fit(data[['X1','X2']],data['y'])
data['Probability'] = svc.predict_proba(data[['X1','X2']])[:,0]
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'],data['X2'],s=30,c=data['Probability'],cmap='Reds')
plt.show()
