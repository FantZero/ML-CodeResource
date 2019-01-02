# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:12:59 2018
参考网址：
    http://www.dmlearning.cn/single/6787c6e70d0940c49883de41ec3d046f.html
    https://zhuanlan.zhihu.com/p/34534004
@author: jz
"""
import numpy as np
import progressbar

from decision_tree.decision_tree_model import DecisionTree
from utils.misc import bar_widgets


class LeastSquaresLoss():
    
    def gradient(self, actual, predicted):      #残差平方和的一阶导数
        return actual - predicted
    
    def hess(self, actual, predicted):      #残差平方和的二阶导数
        return np.ones_like(actual)

class XGBoostRegressionTree(DecisionTree):
    
    def _split(self, y):
        '''y在中列的左半部分包含y_true，在右半部分包含y_pred。拆分并返回两个矩阵'''
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred
    
    #计算切分后的数据集的gain：Gi^2/(Hj+λ)，此处正则化参数λ省略没写
    def _gain(self, y, y_pred):
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y,y_pred).sum()
        return 0.5 * (nominator / denominator)
    
    #通过调用gain()来计算树节点的纯度Gain，并以此来作为树是否分割的标准
    def _gain_by_taylor(self, y, y1, y2):
        y, y_pred = self._split(y)      #分割之前的节点的分数
        y1, y1_pred = self._split(y1)   #分割之后的左子树分数
        y2, y2_pred = self._split(y2)   #分割之后的右子树分数
        
        true_gain = self._gain(y1,y1_pred)
        false_gain = self._gain(y2,y2_pred)
        gain = self._gain(y,y_pred)
        return true_gain + false_gain - gain
    
    #叶节点的具体取值:wj=Gi/(Hj+λ)
    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        gradient = np.sum(self.loss.gradient(y,y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y,y_pred), axis=0)
        update_approximation = gradient / hessian
        return update_approximation
    
    def fit(self, X, y):
        '''将gain_by_taylor()作为切割树的标准，将approximate_update()作为估算子节点取值的方法
            传递给decisionTree，并以此来构建决策树'''
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)        #调用DecisionTree.fit方法进行训练
        
class XGBoost(object):
    '''
    n_estimator: int 树的数量
    learning_rate: float 梯度下降的学习率
    min_samples_split: int 每棵子树的节点的最小数目（小于后不继续切割）
    min_impurity: float 每棵子树的最小纯度（小于后不继续切割）
    max_depth: int 每棵子树的最大层数（大于后不继续切割）
    '''
    
    #构建一个含有n_estimators棵XGBoostRegressionTree的类
    def __init__(self, n_estimators=200, learning_rate=0.01, min_samples_split=2, min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
        self.loss = LeastSquaresLoss()
        
        self.trees = []     #决策树森林
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss)
            self.trees.append(tree)

    def fit(self, X, y):
        m = X.shape[0]
        y = np.reshape(y,(m,-1))
        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y,y_pred), axis=1)
            tree.fit(X, y_and_pred)     #调用XGBoostRegressionTree.fit方法进行训练
            update_pred = tree.predict(X)   #调用DecisionTree.predict方法进行训练
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred += update_pred
            
    def predict(self, X):
        y_pred = None
        m = X.shape[0]
        for tree in self.trees:
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred
        return y_pred
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     
        
        
        
        
        
        