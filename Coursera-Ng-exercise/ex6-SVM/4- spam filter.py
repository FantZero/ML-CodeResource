# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:54:14 2018
4-垃圾邮件检测
@author: jz
"""

from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio

mat_tr = sio.loadmat('./data/spamTrain.mat')
X,y = mat_tr.get('X'),mat_tr.get('y').ravel()
mat_test = sio.loadmat('./data/spamTest.mat')
test_X,test_y = mat_test.get('Xtest'),mat_test.get('ytest').ravel()

#训练SVM模型
#svc = svm.SVC()
#svc.fit(X,y)
#pred = svc.predict(test_X)
#print svc.score(test_X,test_y)
#print metrics.classification_report(test_y,pred)

#训练LR模型
logit = LogisticRegression()
logit.fit(X,y)
pred = logit.predict(test_X)
print logit.score(test_X,test_y)
print metrics.classification_report(test_y,pred)