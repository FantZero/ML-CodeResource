# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:19:13 2018
    
@author: jz
"""
import numpy as np

globalErr = 0;
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)        #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    '''
    数据集合、待切分的特征、该特征的某个值
    在给定特征和特征值确定的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回
    '''
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]     #返回的是一行行的数据组成的子集
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    '''
    负责生成叶节点，当chooseBestSplit()函数确定不再对数据进行切分时，调用该函数来得到叶节点的模型
    在回归树中该模型其实就是目标变量的均值
    '''
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    '''
    误差估计函数：总方差为方差乘以数据集中样本的个数
    '''
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def linearSolve(dataSet):
    '''
    线性处理：将数据集格式化成目标变量Y和自变量X
    '''
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    XTX = np.dot(X.T,X)
    if np.linalg.det(XTX) == 0.0:   #计算行列式值是否为0，即判断是否可逆
        raise NameError('This matrix is singular, cannot do inverse, \n\
                        try increasing the second value of ops')
    ws = np.dot(XTX.I,np.dot(X.T,Y))    #利用正规方程求解回归系数
    return ws,X,Y

def modelLeaf(dataSet):
    '''
    模型树的叶节点模型
    '''
    ws,X,Y = linearSolve(dataSet)
    return ws    
    
def modelErr(dataSet):
    '''
    模型树的误差计算函数
    '''
    ws,X,Y = linearSolve(dataSet)
    yHat = np.dot(X,ws)
    return sum(np.power(Y - yHat,2))

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    回归树构建的核心函数，函数目的是找到数据的最佳二元切分方式，返回切分特征和特征值
    如果找不到一个“好”的二院切分，将函数返回None并同时调用createTree()来产生叶节点，叶节点的值也将返回None
    '''
    tolS = ops[0]   #容许的误差下降值
    tolN = ops[1]   #切分的最少样本点，ops是用户指定的参数，用于控制函数的停止时机
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:  #如果该数目为1，那么就不需要再切分直接返回
        return None,leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):    #遍历所有的特征
        for splitVal in set((dataSet[:,featIndex].T.tolist())[0]):  #遍历该特征下所有的特征值（set集合不重复）
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:      #如果误差减少不大则退出
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):    #如果切分后某个子集的大小过小则不进行切分
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    leafType给出建立叶节点的函数、errorType代表误差计算函数、ops为用户指定参数
    如果是回归树该模型是一个常数；如果是模型树该模型是一个线性方程
    '''
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:return val      #满足停止条件时返回叶节点值
    retTree = {}
    retTree['spInd'] = feat     #切分特征的index
    retTree['spVal'] = val      #切分特征的特征值
    lSet,rSet = binSplitDataSet(dataSet,feat,val)   #左子树 > 切分值，右子树 <= 切分值
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def isTree(obj):
    '''
    判断当前处理的节点是否是叶节点
    '''
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    '''
    递归函数，从上往下遍历树直到叶节点为止，找到两个叶节点则计算它们的平均值。
    '''
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree,testData):
    '''
    tree为待剪枝的树、testData为剪枝所需的测试数据
    '''
    global globalErr
    if np.shape(testData)[0] == 0: return getMean(tree)     #没有测试数据则对树进行塌陷处理（即返回平均值）
   
    #左右分支存在子树，（继续）进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'],rSet)
    
    #两个分支均为叶节点，可以进行合并。要求：合并后的误差比不合并的误差小则进行合并操作，反之不合并直接返回
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            sum(np.power(rSet[:,-1] - tree['right'],2))                 #不合并的误差
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))         #合并的误差
        if errorMerge < errorNoMerge:
#            globalErr += errorMerge
            print ('Merging...')
            return treeMean
        else: return tree
    else: return tree
   

############用树回归进行测试的代码
def regTreeEval(model,inData):
    '''
    回归树的叶节点为float型常量，对于回归树近似乎没有做数值的改变
    '''
    return float(model)

def modelTreeEval(model,inData):
    '''
    模型树的叶节点为浮点型参数的线性方程
    '''
    n = np.shape(inData)[1]
    X = np.mat(np.ones((1,n+1)))    #构建1x(n+1)的单行矩阵
    X[:,1:n+1] = inData             #第一列设置为1，线性方程偏置项b
    return float(X * model)         #返回浮点型的回归系数向量

def treeForeCast(tree,inData,modelEval=regTreeEval):
    '''
    树预测    
    @tree；树回归模型
    @inData：输入数据
    @modelEval：叶节点生成类型，需指定，默认回归树类型
    '''
    if not isTree(tree): return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:   #非叶节点，大于切分值的左子树
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:       #非叶节点，小于等于切分值的右子树
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'],inData)
        
def createForeTree(tree,testData,modelEval=regTreeEval):
    '''
    创建预测树，对测试数据根据modelEval引用的预测函数，进行回归树或模型树的预测
    '''
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat
    
#回归树
#myMat = np.mat(loadDataSet('./ex2.txt'))
#myTree = createTree(myMat,ops=(0,1))
#print myTree
#myMatTest = np.mat(loadDataSet('ex2test.txt'))
#print prune(myTree,myMatTest)

#模型树
myMat2 = np.mat(loadDataSet('exp2.txt'))
print (createTree(myMat2,modelLeaf,modelErr,(1,10)))

#trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
#testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
#创建并测试一棵回归树
#myTree = createTree(trainMat,ops=(1,20))
#yHat = createForeTree(myTree,testMat[:,0])
#print np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
#创建并测试一棵模型树
#myTree = createTree(trainMat,modelLeaf,modelErr,ops=(1,20))
#yHat = createForeTree(myTree,testMat[:,0],modelTreeEval)
#print np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]      #计算yHat与实际结果的皮尔森相关性














