# -*- coding: utf-8 -*-
"""
Created on Mon Nov 05 15:37:44 2018

@author: jz
"""
import numpy as np

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):   #寻找任何一个不等于i的j
    j = i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #根据约束条件求alpha2的值
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    '''
    简化版SMO算法
    数据集、类别标签、常数C、容错率、退出前最大的循环次数
    '''
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    ite = 0
    while(ite < maxIter):
        alphaPairsChanged = 0   #用于记录alpha是否已经进行优化
        for i in range(m):
            #预测的类别，没有引入核函数
            fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C) or (labelMat[i]*Ei > toler) and alphas[i] > 0):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H: print ('L==H');continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T \
                        - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0: print ('eta>=0');continue
                alphas[j] -= labelMat[j] * (Ei-Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(abs(alphas[j] - alphaJold) < 0.00001):print ('j not moving enough');continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if(0 < alphas[i]) and (C > alphas[i]): b = b1
                elif(0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1+b2) / 2.0
                alphaPairsChanged += 1
                print ('iter: %d i :%d, pairs changed %d' %(ite,i,alphaPairsChanged))
        if(alphaPairsChanged == 0): ite += 1
        else: ite = 0
        print ('iteration number: %d' % ite)
    return b,alphas

def kernelTrans(X, A, kTup):
    '''
    转换核函数
    kTup是一个元组，第一个参数是描述所用核函数类型的一个字符串，第二个是σ
    '''
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T
    elif kTup[0] == 'rbf':      #径向基核函数
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    '''
    建立一个数据结构来保存所有重要的值
    '''
    def __init__(self,dataMatIn,classLabels,C,toler, kTup):
        self.X = dataMatIn  #输入变量
        self.labelMat = classLabels     #类标签
        self.C = C      #惩罚参数C
        self.tol = toler    #
        self.m = np.shape(dataMatIn)[0]     #样本容量
        self.alphas = np.mat(np.zeros((self.m,1)))      #拉格朗日乘子α (m,1)
        self.b = 0      #截距项b
        self.eCache = np.mat(np.zeros((self.m,2)))  #第一列表示eCache是否有效标志位，第二列给出实际E值
        self.K = np.mat(np.zeros((self.m, self.m)))     #Σ K(x,xi)的值 (m,m)
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS,k):
    '''
    对于给定的alpha值，计算E值并返回：Ek = fXk - yk
    '''
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T * (oS.K[:,k]) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    '''
    选择第二个alpha(内循环):|Ei - Ek|最大
    '''
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]   #构建一个非零表，返回非零E值对应的alpha值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #选择其中使得改变最大的那个值
            if k == i: continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    '''
    计算误差值并存入缓存中，在对alpha值进行优化之后会用到这个值
    '''
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):
    '''
    选择第二个alpha，并在可能时对其进行优化处理。如果有一对alpha值发生变化，返回1。
    '''
    Ei = calcEk(oS,i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):    #alpha可以进入优化过程
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] - oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j]-oS.alphas[i])
        if L == H: print ('L==H'); return 0
#        eta = 2.0*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        eta = oS.K[i,i] + oS.K[j,j] - 2.0*oS.K[i,j]
        if eta <= 0: print ('eta<=0'); return 0
        oS.alphas[j] += oS.labelMat[j] * (Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)  #剪辑后alpha2的值
        updateEk(oS,j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001): print ('j not moving enough'); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold-oS.alphas[j])  #有α2_new求的α1_new
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i]-alphaIold) * oS.K[i,i] - oS.labelMat[j] \
            * (oS.alphas[j]-alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i]-alphaIold) * oS.K[i,j] - oS.labelMat[j] \
            * (oS.alphas[j]-alphaJold) * oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1+b2) / 2.0  #如果α1_new，α2_new是0或者C，选择他们的中点作为b_new
        return 1
    else: return 0
    
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    '''
    完整版Platt SMO算法
    '''
    oS =optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    ite = 0
    entireSet = True; alphaPairsChanged = 0
    while(ite < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print ('fullSet, iter: %d i:%d,pairs changed %d' %(ite,i,alphaPairsChanged))
            ite +=1
        else:           #遍历非边界的值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ('non-bound,iter: %d i :%d,pairs changed %d' %(ite,i,alphaPairsChanged))
            ite += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print ('iteration number: %d' %ite)
    return oS.b,oS.alphas
                
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w

def testRbf(k1 = 1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]    #支持向量矩阵
    labelSV = labelMat[svInd]
    print ('there are %d Support Vectors' % np.shape(sVs)[0])
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b     #样本i的预测值
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print ('the training error rate is: %f' % (float(errorCount)/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print ('the test error rate is: %f' % (float(errorCount)/m))

#dataArr,labelArr = loadDataSet('testSet.txt')
#b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
#ws = calcWs(alphas, dataArr, labelArr)

testRbf()





















