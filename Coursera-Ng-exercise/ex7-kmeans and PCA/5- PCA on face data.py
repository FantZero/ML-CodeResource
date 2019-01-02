# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 19:27:05 2018
    PCA用于面部数据
@author: jz
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook",style="white")
import numpy as np
import pandas as pd
import scipy.io as sio

import PCA_2D as pca

def get_X(df):
    ones = pd.DataFrame({'ones':np.ones(len(df))})
    data = pd.concat([ones,df],axis=1)
    return data.iloc[:,:-1].as_matrix()

def get_y(df):
    return np.array(df.iloc[:,-1])

def normalize_feature(df):
    return df.apply(lambda column:(column - column.mean()) / column.std())

def plot_n_image(X,n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))
    first_n_image = X[:n,:]
    fig,ax_array = plt.subplots(nrows=grid_size,ncols=grid_size,sharey=True,sharex=True,figsize=(8,8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r,c].imshow(first_n_image[grid_size * r + c].reshape((pic_size,pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

mat = sio.loadmat('./data/ex7faces.mat')
X = np.array([x.reshape((32,32)).T.reshape(1024) for x in mat.get('X')])
#plot_n_image(X,n=64)
#plt.show()

#降到k=100维
U,S,V = pca.pca(X)
Z = pca.project_data(X,U,k=100)
#plot_n_image(Z,n=64)
#plt.show()

#从k=100维升回来
X_recover = pca.recover_data(Z,U)
plot_n_image(X_recover,n=64)
plt.show()

#------------使用sklearn PCA
#from sklearn.decomposition import PCA
#sk_pca = PCA(n_components=100) 
#Z = sk_pca.fit_transform(X)
#plot_n_image(Z,n=64)
#plt.show()
#
#X_recover = sk_pca.inverse_transform(Z)
#plot_n_image(X_recover,n=64)
#plt.show()










