# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:35:48 2018
    在本练习中，我们将使用高斯模型实现异常检测算法，并将其应用于检测网络上的故障服务器。 
    我们还将看到如何使用协作过滤构建推荐系统，并将其应用于电影推荐数据集。
@author: jz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import stats


###################异常检测算法###################
##创建一个返回每个要素的均值和方差的函数。
#def estimate_gaussian(X):
#    mu = X.mean(axis=0)
#    sigma = X.var(axis=0)   #X沿y轴的方差
#    return mu,sigma
#
##找到给定概率密度值和真实标签的最佳阈值。
#def select_threshold(pval,yval):
#    best_epsilon = 0
#    best_f1 = 0
#    f1 = 0
#    
#    step = (pval.max()-pval.min()) / 1000
#    
#    for epsilon in np.arange(pval.min(),pval.max(),step):
#        preds = pval < epsilon
#        tp = np.sum(np.logical_and(preds==1,yval==1)).astype(float)
#        fp = np.sum(np.logical_and(preds==1,yval==0)).astype(float)
#        fn = np.sum(np.logical_and(preds==0,yval==1)).astype(float)
#        
#        precision = tp / (tp+fp)
#        recall = tp / (tp+fn)
#        f1 = (2 * precision * recall) / (precision+recall)
#        
#        if f1 > best_f1:
#            best_f1 = f1
#            best_epsilon = epsilon
#    return best_epsilon,best_f1
#
#data = loadmat('./data/ex8data1.mat')
#X = data['X']
#
#fig,ax = plt.subplots(figsize=(12,8))
#ax.scatter(X[:,0],X[:,1])
#plt.show()
#
#mu,sigma = estimate_gaussian(X)
#
#Xval = data['Xval']
#yval = data['yval']
#
##dist = stats.norm(mu[0],sigma[0])       #训练p(x)函数模型
#
##我们计算并保存给定上述的高斯模型参数的数据集中每个值的概率密度。
#p = np.zeros((X.shape[0],X.shape[1]))
##p[:,0] = stats.norm(mu[0],sigma[0]).pdf(X[:,0])
##p[:,1] = stats.norm(mu[1],sigma[1]).pdf(X[:,1])
#p = stats.norm(mu,sigma).pdf(X)     #X的正态分布
#
#"""
#    我们还需要为验证集（使用相同的模型参数）执行此操作。 我们将使用与真实标签组合的这些概率来确定将数据点
#分配为异常的最佳概率阈值
#"""
#pval = np.zeros((Xval.shape[0],Xval.shape[1]))
##pval[:,0] = stats.norm(mu[0],sigma[0]).pdf(Xval[:,0])
##pval[:,1] = stats.norm(mu[1],sigma[1]).pdf(Xval[:,1])
#pval = stats.norm(mu,sigma).pdf(Xval)
#
#epsilon,f1 = select_threshold(pval,yval)
#
##被认为是离群值的值的索引
#outliers = np.where(p < epsilon)
#
#fig,ax = plt.subplots(figsize=(12,8))
#ax.scatter(X[:,0],X[:,1])
#ax.scatter(X[outliers[0],0],X[outliers[0],1],s=50,color='r',marker='o')
#plt.show()

###################协同过滤算法###################
from scipy.optimize import minimize
def cost(params,Y,R,num_features,learning_rate):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    
    #将参数数组重塑为参数矩阵
    X = np.matrix(np.reshape(params[:num_movies * num_features],(num_movies,num_features))) #(1682,10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:],(num_users,num_features))) #(943,10)
    
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    error = np.multiply((X * Theta.T) - Y,R)
    squared_error = np.power(error,2)
    J = (1. / 2) * np.sum(squared_error)
    
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta,2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X,2)))
    
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)
    
    grad = np.concatenate((np.ravel(X_grad),np.ravel(Theta_grad)))
    
    return J,grad

data = loadmat('./data/ex8_movies.mat')

#Y是包含从1到5的等级的（数量的电影x数量的用户）数组.R是包含指示用户是否给电影评分的二进制值的“指示符”数组。 
#两者具有相同的维度
Y = data['Y']
R = data['R']

#我们还可以通过将矩阵渲染成图像来尝试“可视化”数据。 我们不能从这里收集太多，但它确实给我们了解用户和电影的相对密度
fig,ax = plt.subplots(figsize=(12,12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()

movie_idx = {}
f = open('./data/movie_ids.txt',encoding='gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])
ratings = np.zeros((1682,1))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

R = data['R']
Y = data['Y']
Y = np.append(Y,ratings,axis=1)
R = np.append(R,ratings!=0,axis=1)
movies = Y.shape[0]
users  = Y.shape[1]
features = 10
learning_rate = 10

X = np.random.random(size=(movies,features))
Theta = np.random.random(size=(users,features))
params = np.concatenate((np.ravel(X),np.ravel(Theta)))

Ymean = np.zeros((movies,1))
Ynorm = np.zeros((movies,users))

for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

fmin = minimize(fun=cost,x0=params,args=(Ynorm,R,features,learning_rate),
                method='CG',jac=True,options={'maxiter':100})

X = np.matrix(np.reshape(fmin.x[:movies * features],(movies,features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:],(users,features)))
predictions = X * Theta.T
my_preds = predictions[:,-1] + Ymean
sorted_preds = np.sort(my_preds,axis=0)[::1]
idx = np.argsort(my_preds,axis=0)[::1]
print ("Top 10 movies predictions:")
for i in range(10):
    j = int(idx[i])
    print ("Predicted rating of {0} for movie {1}.".format(str(float(my_preds[j])),movie_idx[j]))













