# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:32:05 2018
寻找最优参数
@author: jz
"""

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex6data3.mat')
training = pd.DataFrame(mat.get('X'),columns=['X1','X2'])   #训练集
training['y'] = mat.get('y')
cv = pd.DataFrame(mat.get('Xval'),columns=['X1','X2'])      #交叉验证集
cv['y'] = mat.get('yval')

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
combination = [(C,gamma) for C in candidate for gamma in candidate]

#手工网格搜索
#search = []
#for C,gamma in combination:
#    svc = svm.SVC(C=C,gamma=gamma)
#    svc.fit(training[['X1','X2']],training['y'])    #训练使用训练集
#    search.append(svc.score(cv[['X1','X2']],cv['y']))   #验证使用交叉验证集
#best_score = search[np.argmax(search)]
#best_param = combination[np.argmax(search)]     #最佳参数配合
#best_svc = svm.SVC(C=best_param[0],gamma=best_param[1])
#best_svc.fit(training[['X1','X2']],training['y'])
#ypred = best_svc.predict(cv[['X1','X2']])
#print 'best_score:',best_score
#print metrics.classification_report(cv['y'],ypred)

#sklearn网格搜素
parameters = {'C':candidate,'gamma':candidate}
svc = svm.SVC()
clf = GridSearchCV(svc,parameters,n_jobs=-1)
clf.fit(training[['X1','X2']],training['y'])
ypred = clf.predict(cv[['X1','X2']])
print (metrics.classification_report(cv['y'],ypred))


