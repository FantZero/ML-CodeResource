# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:48:12 2018
    http://www.dmlearning.cn/single/a5bf33e7b2c44e499a1cb7b2d5f8fbfa.html
@author: jz
"""

from __future__ import division
import numpy as np
import progressbar

from utils import train_test_split, standardize, to_categorical
from utils import mean_squared_error, accuracy_score
from utils.loss_functions import SquareLoss, CrossEntropy, SotfMaxLoss
from decision_tree.decision_tree_model import RegressionTree
from utils.misc import bar_widgets

class GBDT(object):
    '''
    Parameter:
    -----------
    n_estimators: int 树的数量
    learning_rate: float 梯度下降的学习率
    min_samples_split: int 每棵子树的节点的最小数目（小于后不继续切割）
    min_impurity: float 每棵子树的最小纯度（小于后不继续切割）
    max_depth: int 每棵子树的最大层数（大于后不继续切割）
    regression: boolean  是否为回归问题
    '''
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
        self.loss = SquareLoss()    #回归树残差 -(y - y_pred)
        if not self.regression:     #分类树残差 y - p
            self.loss = SotfMaxLoss()
        
        #分类问题也使用回归树，利用残差去学习概率
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                             min_impurity=self.min_impurity,
                                             max_depth=self.max_depth))
    
    '''主要搞懂这个方法，至于回归树的拟合回归之前CART树拟合的知识点'''
    def fit(self, X, y):
        # 让第一棵树去拟合模型
        self.trees[0].fit(X,y)
        y_pred = self.trees[0].predict(X)
        #for循环的过程就是不断让下一棵树拟合上一颗树的"残差"(梯度)。
        for i in self.bar(range(1, self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)        #残差
            self.trees[i].fit(X, gradient)
            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))
            
    def predict(self, X):
        y_pred = self.trees[0].predict(X)
        #for循环的过程就是汇总各棵树的残差得到最后的结果
        for i in range(1, self.n_estimators):   #回归树预测
            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))
            
        if not self.regression:     #分类树预测
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            #将标签设置为概率最大化的值
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GBDTRegressor,self).__init__(n_estimators = n_estimators,
             learning_rate=learning_rate, min_samples_split=min_samples_split,
             min_impurity=min_var_red,max_depth=max_depth,regression=True)
        
class GBDTClassifier(GBDT):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=4, debug=False):
        super(GBDTClassifier,self).__init__(n_estimators = n_estimators,
             learning_rate=learning_rate, min_samples_split=min_samples_split,
             min_impurity=min_info_gain,max_depth=max_depth,regression=False)
    def fit(self, X, y):
        y = to_categorical(y)
        super(GBDTClassifier, self).fit(X,y)

















