# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:41:43 2018

@author: jz
"""
import numpy as np
#import adaboost

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较来对数据进行分类
    数据集、维度、阈值、不等符号
    '''
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':      #将其'<='或 '>'threshVal的数据集标记为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    '''
    遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
    input:数据集、标签集、权值
    return:字典(单层决策树)、错误率、类别估计值
    '''
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):      #第一层循环在数据集的所有特征上遍历
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):    #第二层循环在最大最小之间所有步长值上遍历
            for inequal in ['lt','gt']:     #第三层循环在大于和小于之间切换不等式，选择不等号
                threshVal = (rangeMin + float(j) * stepSize)    #阈值可以设置为整个取值范围之外
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
#                print 'split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' \
#                    %(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i    #分裂的维度
                    bestStump['thresh'] = threshVal     #阈值
                    bestStump['ineq'] = inequal         #不等式
    return bestStump,minError,bestClassEst

#dataMat,classLabels = adaboost.loadSimpData()
#D = np.mat(np.ones((5,1)) / 5)        
#print buildStump(dataMat,classLabels,D)




















