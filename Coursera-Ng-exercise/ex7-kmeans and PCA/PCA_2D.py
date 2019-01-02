# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:52:37 2018
    二维的PCA
@author: jz
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook",style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

        
#X的协方差矩阵
def covariance_matrix(X):
    m = X.shape[0]
    return (np.dot(X.T,X)) / m

#均值归一化
def normalize(X):
    X_copy = X.copy()
    m,n = X_copy.shape
    for col in range(n):        #ndarray.mean()——均值，ndarray.std()——方差
        X_copy[:,col] = (X_copy[:,col] - X_copy[:,col].mean()) / X_copy[:,col].std()
    return X_copy

def pca(X):
    X_norm = normalize(X)   #T1、均值归一化
    Sigma = covariance_matrix(X_norm)   #T2、计算协方差矩阵
    U,S,V = np.linalg.svd(Sigma)    #T3、奇异值分解
    return U,S,V

#利用主成分（矩阵U），我们可以用这些来将原始数据投影到一个较低维的空间中，k表示低维的维度，返回Z(k,1)
def project_data(X,U,k):
    m,n = X.shape
    if k>n:
        raise ValueError('k should be lower dimension of n')
    return np.dot(X,U[:,:k])

#通过反向转换步骤来恢复原始数据
def recover_data(Z,U):
    m,n = Z.shape
    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U,you should recover from lower dimension to higher')
    return np.dot(Z,U[:,:n].T)
    
mat = sio.loadmat('./data/ex7data1.mat')
X = mat.get('X')

#sns.lmplot('X1','X2',data = pd.DataFrame(X,columns=['X1','X2']),fit_reg=False)  #X未均值归一化
#plt.show()

X_norm = normalize(X)
U,S,V = pca(X)
Z = project_data(X_norm,U,1)


#
#fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,4))
#sns.regplot('X1','X2',data=pd.DataFrame(data=X_norm,columns=['X1','X2']),fit_reg=False,ax=ax1)
#ax1.set_title('Original dimension')
#sns.rugplot(Z,ax=ax2)
#ax2.set_xlabel('Z')
#ax2.set_title('Z dimension')
#plt.show()
    
# 将数据恢复到原始维度
X_recover = recover_data(Z,U)
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,4))
 
sns.rugplot(Z,ax=ax1)
ax1.set_title('Z dimension')
ax1.set_xlabel('Z')
 
sns.regplot('X1','X2',
             data = pd.DataFrame(X_recover,columns=['X1','X2']),
             fit_reg=False,
             ax=ax2)
ax2.set_title('2D projection from Z')

sns.regplot('X1','X2',
            data=pd.DataFrame(data=X_norm,columns=['X1','X2']),
            fit_reg=False,
            ax=ax3)
ax3.set_title('Original dimension')
plt.show()
 













   
    
    
    
    
    