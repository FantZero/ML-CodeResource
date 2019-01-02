# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:11:04 2018

@author: jz
"""
import numpy as np

#创建一个包含在所有文档中出现的不重复词汇的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)     #操作符"|"用于求两个集合的并集
    return list(vocabSet)

#输入为词汇表及某个文档，输出是文档向量，向量的每一个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)    #创建一个和词汇表等长的向量，并将其元素都设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ('the word : %s is not in my Vocabulary!' % word)
    return returnVec

#朴素贝叶斯词袋模型，每个单词可以出现多次，与setOfWords2Vec函数相对应
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1    #与上面函数唯一不同的是每遇到一个词增加词向量中的对应值
    return returnVec

#朴素贝叶斯分类器训练函数------核心函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)     #训练集个数
    numWords = len(trainMatrix[0])      #每个训练集所包含的词数，也可以理解成向量的维度
    pAbusive = sum(trainCategory) / float(numTrainDocs)     #求p(Y=C1) 先验概率
    
    #极大似然估计
#    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
#    p0Denom = 0.0; p1Denom = 0.0
    
    #贝叶斯估计
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)    #(类)条件概率分子部分
    p0Denom = 2.0; p1Denom = 2.0        #(类)条件概率分母部分，因为每个属性只有0/1这2种取值，所以为2
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
#    p1Vect = p1Num / p1Denom
#    p0Vect = p0Num / p0Denom
    
    #做对数化操作便于sum时内部连乘，以及相加时与别的对数相乘
    p1Vect = np.log(p1Num / p1Denom)        #y=1时对应的条件概率(所有属性的条件概率，为向量)，向量长度=numWords
    p0Vect = np.log(p0Num / p0Denom)        #y=0时对应的条件概率
    
    return p0Vect,p1Vect,pAbusive

#listOPosts,listClasses = loadDataSet()
#myVocabList = createVocabList(listOPosts)
#trainMat = []
#for postinDoc in listOPosts:
#    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#p0v,p1v,pAb = trainNB0(trainMat,listClasses)

#分类，
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #sum(...)这部分本身就是对数，加上np.log(...)就是相乘
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

'''
    示例：使用贝叶斯进行文本分类
'''
 #收集数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    return postingList,classVec       

def testingNB():
    listOposts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    trainMat = []
    for postinDoc in  listOposts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry ,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry ,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))


'''
    示例：使用朴素贝叶斯过滤垃圾邮件
'''
#接收一个大字符串并将其解析为字符串列表
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    #docList为文档列表，classlist为类别列表，fullText包含所有文档的所有词汇
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('./email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet = []
    
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:    #随机构件训练集和测试集
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    
    for docIndex in testSet:    #对测试集分类
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1     #利用训练集得出的概率对测试数据进行判断，统计错误个数
    print ('the error rate is: ',float(errorCount) / len(testSet))


#testingNB()
spamTest()















