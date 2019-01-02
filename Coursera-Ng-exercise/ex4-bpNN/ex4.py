# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:51:19 2018
    我们将再次处理手写数字数据集，这次使用反向传播的前馈神经网络。 我们将通过反向传播算法
实现神经网络成本函数和梯度计算的非正则化和正则化版本。 我们还将实现随机权重初始化和使用网络进行预测的方法。
    BP算法参考网址：https://www.zhihu.com/question/27239198?rf=24827633
@author: jz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1/(1+np.exp(-z))

#sigmoid的梯度函数，理解成求导函数
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

#前向传播函数
def forward_propagate(X,theta1,theta2):
    m = X.shape[0]
    
    a1 = np.insert(X,0,values=np.ones(m),axis=1)
    z2 = a1*theta1.T
    a2 = np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3 = a2*theta2.T
    h = sigmoid(z3)
    
    return a1,z2,a2,z3,h

#代价函数
#def cost(params,input_size,hidden_size,num_labels,X,y,regParam):
#    m = X.shape[0]
#    X = np.matrix(X)
#    y = np.matrix(y)
#    
#    #将参数数组解开为每个层的参数矩阵
#    theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
#    theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
#    
#    #执行前向传播函数
#    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
#    
#    #计算代价函数
#    J=0
#    for i in range(m):
#        first_term = np.multiply(-y[i,:],np.log(h[i,:]))
#        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
#        J += np.sum(first_term - second_term)
#    J = J/m
#    
#    #加上正则化项
#    J += (float(regParam) / (2*m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))
#    return J

#现在我们准备好实施反向传播来计算梯度。 由于反向传播所需的计算是代价函数中所需的计算过程，
#我们实际上将扩展代价函数以执行反向传播并返回代价和梯度。
#input_size = 400
#hidden_size = 25
#num_labels = 10
#regParam = 1
def backprop(params,input_size,hidden_size,num_labels,X,y,regParam):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    #将参数数组解开为每个层的参数矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
    
    #执行前向传播函数
    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
    
    J = 0
    delta1 = np.zeros(theta1.shape) #(25,401)
    delta2 = np.zeros(theta2.shape) #(10,26)

    #计算代价函数
    for i in range(m):
        first_term = np.multiply(-y[i,:],np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)
    J = J/m
    
    #加上正则化项
    J += (float(regParam) / (2*m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))
    
    #实施反向传播！！   遍历每一个用例     d2t,d3t表示第二层和第三层的误差
    for t in range(m):
        a1t = a1[t,:]   # (1, 401)
        z2t = z2[t,:]   # (1, 25)
        a2t = a2[t,:]   # (1, 26)
        ht = h[t,:]     #(1,10)
        yt = y[t,:]     #(1,10)
        
        d3t = ht -yt    #(1,10)
        
        z2t = np.insert(z2t,0,values=np.ones(1))    #(1,26)
        d2t = np.multiply((theta2.T * d3t.T).T,sigmoid_gradient(z2t))   #(1,26)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    #将梯度计算加正则化
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * regParam) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * regParam) / m
    
    #将梯度矩阵分解成单个数组
    grad = np.concatenate((np.ravel(delta1),np.ravel(delta2)))
    
    return J,grad

#import matplotlib.cm as cm
#def plot_hidden_layer(theta):
#    """
#    theta: (10285, )
#    """
#    final_theta1, _ = deserialize(theta)
#    hidden_layer = final_theta1[:, 1:]  # ger rid of bias term theta
#
#    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))
#
#    for r in range(5):
#        for c in range(5):
#            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
#                                   cmap=cm.binary)
#            plt.xticks(np.array([]))
#            plt.yticks(np.array([]))
#
#def deserialize(seq):
##     """into ndarray of (25, 401), (10, 26)"""
#    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)

data = loadmat('ex4data1.mat')

X = data['X']
y = data['y']

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

#初始化设置
input_size = 400
hidden_size = 25 #每个中间层的单元数
num_labels = 10
regParam = 1

#随机初始化完整网络参数大小的参数数组
params = (np.random.random(size=hidden_size * (input_size+1) +  num_labels * (hidden_size+1)) - 0.5) * 0.25
m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

fmin = minimize(fun=backprop,x0=params,args=(input_size,hidden_size,num_labels,X,y_onehot,regParam),
                method='TNC',jac=True,options={'maxiter':250})
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size+1)],(hidden_size,(input_size+1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size+1):],(num_labels,(hidden_size+1))))

a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
y_pred = np.array(np.argmax(h,axis=1)+1)

correct = [1 if a==b else 0 for (a,b) in zip(y_pred,y)]
accuracy = sum(map(int,correct)) / float(len(correct))
print ('accuracy = {0}%'.format(accuracy * 100))




















