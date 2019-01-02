# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:43:01 2018

@author: jz
"""
import numpy as np
import boost

def loadSimpData():
    dataMat = np.matrix([[1, 2.1],
                         [2, 1.1],
                         [1.3, 1],
                         [1, 1],
                         [2, 1]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def loadDataSet(fileName):
    '''
    自适应数据加载函数
    '''
    numFeat = len(open(fileName).readline().split('\t'))    #特征数量+1(类别标签)
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    基于单层决策树的AdaBoost训练过程
    input:数据集、类别标签、迭代次数numIt
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1)) / m)  #初始数据点权重
    aggClassEst = np.mat(np.zeros((m,1)))#数据点的类别估计累计值——基本分类器的线性组合f(x),np.sign(aggClassEst)即为G(x)
    for i in range(numIt):
        bestStump, error, classEst = boost.buildStump(dataArr,classLabels,D)
        print ('D: ',D.T)
        alpha = float(0.5 * np.log((1.0-error) / max(error,1e-16)))   #max(error,1e-16)用于确保在没有错误时无除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print ('classEst: ',classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))    #此时D的结果对应公式中的w_mi*exp(-α_m*y_i*Gm(x_i))
        D = D / D.sum()     #w_m+1，其中D.sum()对应公式中的Zm(规范因子)
        aggClassEst += alpha * classEst
        print ('aggClassEst: ',aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print ('total error: ',errorRate,'\n')
        if errorRate == 0.0:break
    return weakClassArr

def adaClassify(dataToClass,classifierArr):
    '''
    AdaBoost分类函数
    input:一个或多个待分类样例、多个弱分类器组成的数组
    '''
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))   #同上
    for i in range(len(classifierArr)):
        classEst = boost.stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                       classifierArr[i]['thresh'],\
                                       classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print (aggClassEst)
    return np.sign(aggClassEst)
    
    
#dataMat,classLabels = loadSimpData()
#classifierArray = adaBoostTrainDS(dataMat,classLabels,30)
#print adaClassify([0,0],classifierArray)

dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray = adaBoostTrainDS(dataArr,labelArr,10)

testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr,classifierArray)
errArr = np.mat(np.ones((67,1)))
print (errArr[prediction10!=np.mat(testLabelArr).T].sum() / 67)


















