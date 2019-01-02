# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 17:34:54 2018
    
@author: jz
"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook",style="white",palette=sns.color_palette("RdBu"))
import numpy as np
import pandas as pd
import scipy.io as sio
import io

import scipy.optimize as opt

"""
备注:
X - num_movies (1682) X num_features(10)电影特性矩阵
Theta - num_users (943) x num_features(10)用户特性矩阵
Y - num_movies x num_users电影用户评级矩阵
R - num_movies x num_users矩阵，其中R(i, j) = 1如果第i部电影由第j个用户评分
"""

#将电影参数矩阵X 和 用户参数矩阵theta合并成一维序列
def serialize(X,theta):
    return np.concatenate((X.ravel(),theta.ravel()))

#将param拆分成电影参数矩阵X 和 用户参数矩阵theta
def deserialize(param,n_movie,n_user,n_features):
    return param[:n_movie * n_features].reshape(n_movie,n_features), \
        param[n_movie * n_features:].reshape(n_user,n_features)

#
def cost(param,Y,R,n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    n_movie,n_user = Y.shape
    X, theta = deserialize(param,n_movie,n_user,n_features)
    inner = np.multiply(np.dot(X,theta.T)-Y,R)
    return np.power(inner,2).sum() / 2

def gradient(param,Y,R,n_features):
    n_movie,n_user = Y.shape
    X,theta = deserialize(param,n_movie,n_user,n_features)
    inner = np.multiply(np.dot(X,theta.T)-Y,R)
    
    X_grad = np.dot(inner,theta)
    theta_grad = np.dot(inner.T,X)
    return serialize(X_grad,theta_grad)

#带正则化的代价函数
def regularized_cost(param,Y,R,n_features,l=1):
    reg_term = np.power(param,2).sum() * (l / 2)
    return cost(param,Y,R,n_features) + reg_term

#带正则化的梯度下降函数
def regularized_gradient(param,Y,R,n_features,l=1):
    grad = gradient(param,Y,R,n_features)
    reg_term = l * param
    return grad + reg_term


movies_mat = sio.loadmat("./data/ex8_movies.mat")
#Y,R = movies_mat.get('Y'),movies_mat.get('R')
#m,u = Y.shape   #m-电影数量，u-用户数量
#n = 10      #电影特征数

#param_mat = sio.loadmat('./data/ex8_movieParams.mat')
#theta,X = param_mat.get('Theta'),param_mat.get('X') #theta-(943,10);X-(1682,10)

#param = serialize(X,theta)
#n_movie,n_user = Y.shape
#X_grad,theta_grad = deserialize(gradient(param,Y,R,10),n_movie,n_user,10)
#assert X_grad.shape == X.shape
#assert theta_grad.shape == theta.shape

movie_list = []
with io.open('./data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))
movie_list = np.array(movie_list)

ratings = np.zeros(1682)    #有1682部电影
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

Y,R = movies_mat.get('Y'),movies_mat.get('R')
Y = np.insert(Y,0,ratings,axis=1)   #(电影数量,用户数量)
R = np.insert(R,0,ratings != 0,axis=1)  #R中有评分的位置插入1
n_features = 50
n_movie,n_user = Y.shape
l = 10  #正则项
X = np.random.standard_normal((n_movie,n_features))
theta = np.random.standard_normal((n_user,n_features))
param = serialize(X,theta)

#均值归一化处理,将每一个用户对某一部电影的评分减去所有用户对该电影评分的平均值
#Y_norm = Y - Y.mean()
mu = np.matrix(Y.mean(axis=1)).T
Y_norm = np.array(Y - mu)

res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Y_norm,R,n_features,l),
                   method='TNC',
                   jac=regularized_gradient)

X_trained,theta_trained = deserialize(res.x,n_movie,n_user,n_features)
prediction = np.dot(X_trained,theta_trained.T)
my_preds = np.array(prediction + mu)[:,0]       #取第一个（对应前面ratings）用户的所有电影评分向量
idx = np.argsort(my_preds)[::-1]
for m in movie_list[idx][:10]:
    print (m)

















