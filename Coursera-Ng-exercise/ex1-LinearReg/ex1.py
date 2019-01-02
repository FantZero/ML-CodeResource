# -*- coding: utf-8 -*-
"""
Created on Sat May 05 11:57:15 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib as plt

#代价函数
def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

#梯度下降函数，#alpha-学习率，iters-迭代次数
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))     #所有theta的临时存储位置
    parameters = int(theta.ravel().shape[1])    #参数数量
    cost = np.zeros(iters)  #每次迭代的代价误差
    for i in range(iters):
        error = (X*theta.T) - y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])    #这里包括下一步，对θi进行更新
            temp[0,j] = theta[0,j] - ((alpha/len(X)) * np.sum(term))
        theta = temp                            #更新θ
        cost[i] = computeCost(X,y,theta)        #存储每次的θ对应的代价函数值
    return theta,cost

#单变量线性回归------------------
path = 'ex1data1.txt'
data = pd.read_csv(path,header=None,names=['population','profit'])
#data.plot(kind='scatter',x='population',y='profit',figsize=(12,8))     #原始数据分布
data.insert(0,'ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
alpha = 0.01
iters = 1000
g,cost = gradientDescent(X,y,theta,alpha,iters)
print (computeCost(X,y,g))     #最终的代价
x = np.linspace(data.population.min(),data.population.max(),100)
f = g[0,0]+g[0,1]*x     #回归函数

#线性回归可视化
fig,ax = plt.pyplot.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='prediction')
ax.scatter(data.population,data.profit,label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('predict profit by population')
plt.pyplot.show()

#代价函数可视化
fig2,ax2 = plt.pyplot.subplots(figsize=(12,8))
ax2.plot(np.arange(iters),cost,'r')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Cost_Iterations')
plt.pyplot.show()


#多变量线性回归----------------------
path2 = 'ex1data2.txt'
data2 = pd.read_csv(path2,header=None,names=['size','bedrooms','price'])
data2 = (data2-data2.mean())/data2.std()    #特征归一
data2.insert(0,'ones',1)
cols2 = data2.shape[1]
X2 = data2.iloc[:,0:cols2-1]
y2 = data2.iloc[:,cols2-1:cols2]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))
alpha = 0.01
iters = 1000
g2,cost2 = gradientDescent(X2,y2,theta2,alpha,iters)
print (computeCost(X2,y2,g2))     #最终的代价


#正规方程—单变量线性回归-------------
def normalEqn(X,y):
    theta3 = ((np.linalg.inv(np.dot(X.T,X))).dot(X.T)).dot(y)
    return theta3
g3 = normalEqn(X,y)
x3 = np.linspace(data.population.min(),data.population.max(),100)
f3 = g3[0,0]+g3[1,0]*x3
fig3,ax3 = plt.pyplot.subplots(figsize=(12,8))
ax3.plot(x3,f3,'r',label='prediction')
ax3.scatter(data.population,data.profit,label='Traning Data')
ax3.legend(loc=2)
ax3.set_xlabel('population')
ax3.set_ylabel('profit')
ax3.set_title('predict profit by population')
plt.pyplot.show()