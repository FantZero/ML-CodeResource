# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 18:36:58 2018

@author: jz
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io as sio

mat = sio.loadmat('./data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'),columns=['X1','X2'])

def combine_data_C(data,C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c

#从数据集中选随机择k个样本作为init质心
def random_init(data,k):
    return data.sample(k).as_matrix()

#寻找实例的聚类中心
def _find_your_cluster(x,centroids):
    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=centroids - x)
    return np.argmin(distances)

#为数据中的每个节点分配聚类点，返回所有实例对应聚类点索引的数组C
def assign_cluster(data,centroids):
    return np.apply_along_axis(lambda x: _find_your_cluster(x,centroids),   #寻找矩阵data中每个实例的聚类中心
                               axis=1,
                               arr=data.as_matrix())

#获取根据所有实例标记C之后的新的聚类点
def new_centroids(data,C):
    data_with_c = combine_data_C(data,C)
    return data_with_c.groupby('C',as_index=False).mean().sort_values(by='C').drop('C',axis=1).as_matrix()

#代价函数
def cost(data,centroids,C):
    m = data.shape[0]
    expand_C_with_centroids = centroids[C]  #每个实例对应的聚类点坐标集合,大小跟data一致
    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                   axis = 1,
                                   arr = data.as_matrix() - expand_C_with_centroids)
    return distances.sum()/m

#一次k-mean,epoch表示更新聚类点次数，每次都会缩短所有实例到聚类点之间的距离和
def _k_means_iter(data,k,epoch=100,tol=0.0001):
    centroids = random_init(data,k)
    cost_progress = []
    for i in range(epoch):
        print ('running epoch {}'.format(i))
        C = assign_cluster(data,centroids)      #所有实例分类后对应聚类点索引的数组C
        centroids = new_centroids(data,C)       #重新生成聚类点
        cost_progress.append(cost(data,centroids,C))    #对上面两步操作所得的聚类进行代价计算，每次新加的都会减小
        if len(cost_progress) > 1:
            if(np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break   #新加入的误差变化率小于0.0001，则近似看做聚类中心点导致的误差不再变化，停止循环
    return C,centroids,cost_progress[-1]

#取多次随机聚类点群init，多次k-mean，选择代价值最小的的一个返回，n_init代表随机init数目
def k_means(data,k,epoch=100,n_init=10):
    tries = np.array([_k_means_iter(data,k,epoch) for _ in range(n_init)])  #tries['C','centroids','cost']
    least_cost_idx = np.argmin(tries[:,-1])     #tries.shape()为m*3，tries[:,-1]表示所有随机init的cost
    return tries[least_cost_idx]

#只进行一次随机聚类点群init（演示操作）
#final_C,final_centroids,_ = _k_means_iter(data2,3)
#data_with_c = combine_data_C(data2,final_C)
#sns.lmplot('X1','X2',hue='C',data=data_with_c,fit_reg=False)
#plt.show()

#进行多次随机聚类点群init（正常操作）
best_C,best_centroids,least_cost = k_means(data2,3)
data_with_c = combine_data_C(data2,best_C)
sns.lmplot('X1','X2',hue='C',data=data_with_c,fit_reg=False)
plt.show()

#try sklearn kmeans
#from sklearn.cluster import KMeans
#sk_kmeans = KMeans(n_clusters = 3)
#sk_kmeans.fit(data2)
#sk_C = sk_kmeans.predict(data2)
#data_with_c = combine_data_C(data2,sk_C)    
#sns.lmplot('X1','X2',hue='C',data=data_with_c,fit_reg=False)
#plt.show()
    
    
    
    
    
    
    
    
    
    
    