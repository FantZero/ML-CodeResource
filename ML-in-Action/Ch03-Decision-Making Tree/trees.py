# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:29:09 2018
    1.在数据集中度量一致性
    2.使用递归构造决策树（ID3算法）
    3.使用Matplotlib绘制树形图
@author: jz
"""
from math import log
import operator
import treePlotter
import pickle

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算给定数据集的熵
def calcShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:     #计算 Dik（子集Di中属于类Ck的样本的集合为Dik）
        currentLabel = featVec[-1]
        if currentLabel not  in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:     #计算 H(D)
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt
#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #获取数据集的特征数
    baseEntropy = calcShannonEnt(dataSet)       #计算数据集的经验熵 H(D)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #使用列表推导来创建新的列表，存放第i个特征值所有可能的值
        uniqueVals = set(featList)      #set集合得到列表中唯一元素值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)      #根据第i个特征划分出D1k,D2k,D3k...(Dik)
            prob = len(subDataSet) / float(len(dataSet))    #经验条件熵 H(D|A)的左半部分
            newEntropy += prob * calcShannonEnt(subDataSet) #计算得出经验条件熵 H(D|A)
        infoGain = baseEntropy - newEntropy     #计算信息增益 g(D,A)=H(D)-H(D|A)
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  #返回最优特征索引
#多数表决方法决定该叶节点的分类
def majoriyCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not  in classCount.keys() : classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#创建树，使用Python的字典类型存储树的信息
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:        #遍历完所有特征时返回出现次数最多的类别
        return majoriyCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])   #去除标签列表中已经选为最优特征的那个
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)    #得到列表包含的所有属性值
    for value in uniqueVals:
        subLabels = labels[:]       #复制类标签，保证每次调用函数createTree()时不改变原始列表的内容
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
#使用决策树进行分类
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]      #找到当前树的顶点label
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)      #当前树的顶点label对应的index
    for key in secondDict.keys():       #遍历当前顶点的左右子树
        if testVec[featIndex] == key:   #进入对应子节点的子树
            if type(secondDict[key]).__name__ == 'dict':    #子树为树结构
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:       #子树为叶节点，将叶节点所属类赋值给classLabel
                classLabel = secondDict[key]
    return classLabel
#使用pickle模块存储决策树
def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
#使用pickle模块读取决策树存储文件
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

#myData,labels = createDataSet()
#myTree = treePlotter.retrieveTree(0)
#storeTree(myTree,'classifierStorage_DIY.txt')
#grabTree('classifierStorage_DIY.txt')

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)







