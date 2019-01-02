# -*- coding: utf-8 -*-
"""
Created on Sat Jun 02 11:11:55 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.io import loadmat
from scipy.optimize import minimize

#S形函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#正则化代价函数
def cost(theta, X, y, regParam):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (regParam / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

#正则化梯度下降函数，只进行了一次梯度下降，并且进行向量化
def gradient(theta,X,y,regParam):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = sigmoid(X*theta.T)-y
    grad = ((X.T*error)/len(X)).T + ((regParam/len(X)) * theta)
    grad[0,0] = np.sum(np.multiply(error,X[:,0]))/len(X)    #对θ0不做惩罚
    return np.array(grad).ravel()

#训练所有训练集，返回10个数字对应的θ向量all_theta
def one_vs_all(X,y,num_labels,regParam):
    rows = X.shape[0]       #行数，表示训练集实例数量
    params = X.shape[1]     #参数θ个数
    all_theta = np.zeros((num_labels,params+1))     #10x401，多的一列为截距项（常数项）
    X = np.insert(X,0,values=np.ones(rows),axis=1)  #为X添加额外参数1，位置在第一列
    for i in range(1,num_labels+1):     #循环遍历0-9的训练集
        theta = np.zeros(params+1)      #401x1
        y_i = np.array([i if label==i else 0 for label in y])   #与当前label相等的训练样本设为1，其余的都设为0
        y_i = np.reshape(y_i,(rows,1))  #每一个数字有自己对应的y_i，均为5000x1
        #每次传入theta进行训练，X表示训练样本，y_i表示训练样本对应正确结果集，每次有500行值为1
        fmin = minimize(fun=cost,x0=theta,args=(X,y_i,regParam),method='TNC',jac=gradient)
        all_theta[i-1,:] = fmin.x       #讲训练得到的1-10的θ存入all_theta,一行对应一个
    return all_theta

def predict_all(X,all_theta):
    rows = X.shape[0]
    X = np.insert(X,0,values=np.ones(rows),axis=1)
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    h = sigmoid(X*all_theta.T)      #5000x10,正常结果每一列有500行对应的值等于1或很近似1
    h_argmax = np.argmax(h,axis=1)  #5000x1,与data['y']相对应
    h_argmax = h_argmax+1
    return h_argmax


#数据初始化
data = loadmat('ex3data1.mat')
#rows = data['X'].shape[0]
#params = data['X'].shape[1]
#all_theta = np.zeros((10,params+1))
#X = np.insert(data['X'],0,values=np.ones(rows),axis=1)
#theta = np.zeros(params+1)

all_theta = one_vs_all(data['X'],data['y'],10,1)

y_pred = predict_all(data['X'],all_theta)
correct = [1 if a==b else 0 for (a,b) in zip(y_pred,data['y'])]
accuracy = (sum(map(int,correct))/float(len(correct)))
print ('accuracy = {0}%'.format(accuracy*100))







