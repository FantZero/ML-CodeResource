# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:54:02 2018
    1-异常检测
    我们将使用高斯模型实现异常检测算法，并将其应用于检测网络上的故障服务器。
@author: jz
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook",style="white",palette=sns.color_palette("RdBu"))
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score,classification_report

#1、利用训练集XX对多元高斯模型进行建模
#2、使用交叉验证组(Xval yval)，通过寻找最好的f值来寻找最佳ϵ（阈值）
def select_threshold(X,Xval,yval):
    #根据测试集数据，估计特征值平均值和协方差并构建p(x)函数模型：multi_normal
    mu = X.mean(axis=0)
    cov = np.cov(X.T)   #协方差矩阵
    multi_normal = stats.multivariate_normal(mu,cov)    #多元高斯分布
    
    #这是关键，使用CV(Xval,yval)数据微调超参数，后面找出最佳ϵ
    pval = multi_normal.pdf(Xval)   #即Xval通过构建的p(x)计算的到的对应概率密度值
    
    epsilon = np.linspace(np.min(pval),np.max(pval),num=10000)
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval,y_pred))
    argmax_fs = np.argmax(fs)
    return epsilon[argmax_fs],fs[argmax_fs]

#用最优ϵ，结合X, Xval，预测Xtest。
#返回：multi_normal:多元标准模型；y_pred:测试数据的预测
def predict(X,Xval,e,Xtest,ytest):
    Xdata = np.concatenate((X,Xval),axis=0)     #连接X和Xval数组
    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu,cov)
    
    pval = multi_normal.pdf(Xtest)
    y_pred = (pval <= e).astype('int')
    return multi_normal,y_pred



mat = sio.loadmat('./data/ex8data1.mat')
X = mat.get('X')

#将原始验证数据分为验证和测试集
Xval,Xtest,yval,ytest = train_test_split(mat.get('Xval'),
                                         mat.get('yval').ravel(),
                                         test_size=0.5)
#sns.regplot('Latency','Throughput',
#            data=pd.DataFrame(X,columns=['Latency','Throughput']),
#            fit_reg=False,
#            scatter_kws={"s":20,"alpha":0.5})
#plt.show()

#计算均值和协方差矩阵
#mu = X.mean(axis=0)
#cov = np.cov(X.T)
#multi_normal = stats.multivariate_normal(mu,cov)
#x,y = np.mgrid[0:30:0.01,0:30:0.01]
#pos = np.dstack((x,y))
#fig,ax = plt.subplots()
#ax.contourf(x,y,multi_normal.pdf(pos),cmap='Blues')
#sns.regplot('Latency','Throughput',
#            data=pd.DataFrame(X,columns=['Latency','Throughput']),
#            fit_reg=False,
#            ax=ax,
#            scatter_kws={"s":10,"alpha":0.4})
#plt.show()

e,fs = select_threshold(X,Xval,yval)
multi_normal,y_pred = predict(X,Xval,e,Xtest,ytest)
data = pd.DataFrame(Xtest,columns=['Latency','Throughput'])
data['y_pred'] = y_pred
x,y = np.mgrid[0:30:0.01,0:30:0.01]     #0-30,0.01为间隔的切片
pos = np.dstack((x,y))
fig,ax = plt.subplots()
ax.contourf(x,y,multi_normal.pdf(pos),cmap='Blues')
sns.regplot('Latency','Throughput',
            data=data,
            fit_reg=False,
            ax=ax,
            scatter_kws={"s":10,"alpha":0.4})
anamoly_data = data[data['y_pred']==1]      #测试机异常点
ax.scatter(anamoly_data['Latency'],anamoly_data['Throughput'],marker='x',s=50)
plt.show()





















