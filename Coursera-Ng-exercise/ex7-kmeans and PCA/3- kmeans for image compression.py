# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:09:04 2018
    kmeans用于图像压缩
@author: jz
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from skimage import io
#import imp
#kmean = imp.load_source('./kmeans.py')
#import kmean
import kmeans as keman

pic = io.imread('./data/bird_small.png')  / 255.
#io.imshow(pic)
data = pic.reshape(128*128,3)

#k-mean
C,centroids,cost = keman.k_means(pd.DataFrame(data),16,epoch=10,n_init=3)
compressed_pic = centroids[C].reshape((128,128,3))  #a[b]中每个元素为b[i]为索引，对应a中的每行

#sklearn KMeans
#from sklearn.cluster import KMeans
#model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
#model.fit(data)
#centroids = model.cluster_centers_
#C = model.predict(data)
#compressed_pic = centroids[C].reshape((128,128,3))

fix,ax = plt.subplots(1,2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()