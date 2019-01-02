# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:09:13 2018
    使用k-近邻算法改进约会网站的配对效果，没有使用kd树(k决策树)，直接算的所有训练集数据和目标数据x的距离然后排序
@author: jz
"""

import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
    
def classify0(inX,dataSet,labels,k):
    '''
    inX:需要分类的输入向量
    dataSet：输入的训练样本集
    labels：标签向量
    k:选择最近邻的数目
    '''
    dataSetSize = dataSet.shape[0]
    
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet    #将inX复制成dataSetSize行，列不变，然后减去原训练集数组
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    
    sortedDistIndicies = distances.argsort()    #获取距离排序后的数组下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #classCount.get(voteIlabel,0):如果指定键的值不存在时，返回该默认值0.
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    sortedDistIndicies = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    
    return sortedDistIndicies[0][0]

#将文本记录转换成NumPy的解析程序
def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    
    returnMat = np.zeros((numberOfLines,3))
    classLabelVecyor = []
    
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]          #前三列为输入向量空间，最后一列为类标签
        classLabelVecyor.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVecyor

#归一化特征值 newValue=(oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    
    normDataSet = dataSet - np.tile(minVals,(m,1))      #当前值减去最小值
    normDataSet = normDataSet / np.tile(ranges,(m,1))   #然后除以取值范围
    
    return normDataSet,ranges,minVals

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10      #测试数据集占总数据集比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  #选取测试数据结束位置的下标
    errorCount = 0.0
    for i in range(numTestVecs):
        # normMat[i,:]-测试数据，normMat[numTestVecs:m,:]-训练数据，datingLabels[numTestVecs:m]-训练数据类标签
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ('the classifier came back with: %d,the real answer is: %d' %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]): 
            errorCount+=1.0
    print ('the total error rate is:%f' %(errorCount/float(numTestVecs)))

#根据输入特征判断属于那类结果
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input('percentage of time spent playing video games:'))
    ffMiles = float(input('frequent flier miles earned per year:'))
    iceCream = float(input('liters of ice cream sonsumed per year:'))
    
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0(inArr,normMat,datingLabels,3)
    print ('you wil probably like this guy :',resultList[classifierResult - 1])

datingClassTest()







