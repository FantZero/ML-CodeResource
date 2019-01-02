# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:18:34 2018
参考网址：
    https://blog.csdn.net/QcloudCommunity/article/details/79363040
@author: jz
"""
import numpy as np
from random import seed
from random import randrange
from csv import reader
from math import sqrt

#计算一个分割数据集的基尼指数
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:    #各组(D1、D2)Gini指数加权和
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:   #根据每个类别的得分给小组打分
            p = [row[-1] for row in group].count(class_val) / size
            score += p*p    #p1^2+p2^2，p2=1-p1
        gini += (1.0 - score) * (size / n_instances)    #(1.0-score) == 1-(p1^2+p2^2) == 2*p1(1-p1)即Gini(p)
    return gini     #返回最终的Gini(D,A)

#根据属性和属性值（分为小于和大于等于两部分:D1,D2）分割数据集
def test_split(index, value, dataSet):
    left, right = list(), list()
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

#为数据集选择最好的切分点
def get_split(dataSet, n_features):
    class_values = list(set(row[-1] for row in dataSet))    #数据类别集合(无重复)
    b_index, b_value, b_score, b_groups = 999,999,999,None
    features = list()   #随机选择的属性的下标集合
    while len(features) < n_features:
        index = randrange(len(dataSet[0]) - 1)
        if index not in features:
            features.append(index)
    
    #根据随机选择的属性集合，对数据集中的所有对应的随机属性及其属性值，两次内循环
    #找出其Gini系数最小对应的最优特征与最优切分点，以及区分后的数据集D1/D2
    for index in features:
        for row in dataSet:
            groups = test_split(index, row[index], dataSet)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini,groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

#将数据集分成 n_folds份便于交叉验证
def cross_validation_split(dataSet, n_folds):
    dataSet_split = list()
    dataSet_copy = list(dataSet)
    fold_size = int(len(dataSet) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))
        dataSet_split.append(fold)
    return dataSet_split

#计算准确率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.00 

#创建终端节点值，多数表决
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#为节点创建子分割或生成终端
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)

#造一课决策树
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train,n_features)  #根节点
    split(root,max_depth,min_size,n_features,1) #不断增加子树
    return root

#用决策树进行预测
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']

#从数据集中随机创建一个替换子样本
def subsample(dataSet, ratio):
    sample = list()
    n_sample = round(len(dataSet) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataSet))
        sample.append(dataSet[index])
    return sample

#用一列袋装树做一个预测
def bagging_predict(trees, row):
    predictions = [predict(tree,row) for tree in trees]     #随机森林中所有决策树的预测结果
    return max(set(predictions), key = predictions.count)   #通过多数表决的方式决定预测类别

#随机森林算法
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()  #森林
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample,max_depth,min_size,n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees,row) for row in test]
    return (predictions)

#使用交叉验证拆分评估算法
def evaluate_algorithm(dataSet, algorithm, n_folds, *args):
    folds = cross_validation_split(dataSet, n_folds)    #势必要采用交叉验证的手段
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)      #n_folds-1份作为train
        train_set = sum(train_set,[])
        test_set = list()
        for row in fold:            #1份作为test
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)   #所有的预测类别
        actual = [row[-1] for row in fold]      #所有的真实类别
        accuracy = accuracy_metric(actual, predicted)   #计算准确率
        scores.append(accuracy)
    return scores

#加载csv文件
def load_csv(filename):
    dataSet = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataSet.append(row)
    return dataSet

#将字符串列转换成浮点类型
def str_column_to_float(dataSet, column):
    for row in dataSet:
        row[column] = float(row[column].strip())

#将字符串列转换成整数类型
def str_column_to_int(dataSet, column):
    class_values = [row[column] for row in dataSet]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i       #将最后一列(类别)用0/1表示
    for row in dataSet:
        row[column] = lookup[row[column]]
    return lookup

seed(2)
dataSet = load_csv('sonar.all-data.csv')
for i in range(0,len(dataSet[0]) - 1):
    str_column_to_float(dataSet,i)
str_column_to_int(dataSet,len(dataSet[0]) - 1)
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0   #随机样本大小所占比
n_features = int(sqrt(len(dataSet[0]) -1 ))     #随机选择属性的个数
for n_trees in [1,5,10,15]:
    scores = evaluate_algorithm(dataSet,random_forest,n_folds,max_depth,min_size,sample_size,n_trees,n_features)
    print ('Trees: %d' % n_trees)
    print ('Scores: %s' % scores)
    print ('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))









