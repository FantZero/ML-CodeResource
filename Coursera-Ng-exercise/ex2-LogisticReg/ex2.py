# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:29:17 2018

@author: Administrator
"""
#############################逻辑回归#############################
import numpy as np
import pandas as pd
import matplotlib as plt
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

#S形函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#代价函数
def cost(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply(1-y,np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/len(X)

#梯度下降函数，仅计算了一个梯度步长
def gradient(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])    #θ的数量
    grad = np.zeros(parameters)
    error = sigmoid(X*theta.T) - y
    for i in range(parameters):
        term = np.multiply(error,X[:,i])
        grad[i] = np.sum(term)/len(X)       #代表θ每次梯度下降所减的值
    return grad

#预测函数
def predict(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]

#导入数据
path = 'ex2data1.txt'
data = pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
data.head()

#原始数据展示
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
fig,ax = plt.pyplot.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s = 50,c = 'b'
           ,marker = 'o',label = 'Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s = 50,c = 'r'
           ,marker = 'x',label = 'Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#plt.pyplot.show()

#调整数据格式
data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.zeros(3)             #表明函数为：θ0+θ1x1+θ2x2=0，是一条直线

#使用SciPy's truncated newton（TNC）实现寻找最优参数，无需加入学习率α，求出θ
result = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y))

#计算所得θ的正确率
theta_min = np.matrix(result[0])
predictions = predict(theta_min,X)
correct = [1 if ((a==0 and b==0) or (a==1 and b==1)) else 0 for (a,b) in zip(predictions,y)]
accuracy = sum(map(int,correct)) % len(correct)
print ("正确率为{0}%".format(accuracy))

#打印回归函数
x = np.linspace(X[:,1].min()-10,X[:,1].max()+10,50)
f = -(theta_min[0,0]+theta_min[0,1]*x)/theta_min[0,2]       #θ0+θ1x1+θ2x2=0,结果为f=x2,x=x1的函数

ax.plot(x,f,'r')
ax.legend(loc=2)
plt.pyplot.show()           #记住前面的show()方法需要注释掉

#----------------------正则化逻辑回归---------------------使用ex2data2.txt里面的数据
#正则化代价函数
def costReg(theta,X,y,regParam):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg = (regParam/(2*len(X)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))   #θ1-θ10，没有θ0
    return np.sum(first-second)/len(X)+reg

#正则化梯度下降函数，只进行了一次梯度下降
def gradientRag(theta,X,y,regParam):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X*theta.T)-y
    for i in range(parameters):
        term = np.multiply(error,X[:,i])
        if(i ==0 ):         #θ0不作惩罚处理
            grad[i] = np.sum(term)/len(X)
        else:
            grad[i] = np.sum(term)/len(X)+regParam/len(X)*theta[:,i]
    return grad
    
path = 'ex2data2.txt'
data2 = pd.read_csv(path,header=None,names=['Test 1','Test 2','Accepted'])

#原始数据展示
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
fig,ax = plt.pyplot.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'],positive['Test 2'],s = 50,c = 'b'
           ,marker = 'o',label = 'Accepted')
ax.scatter(negative['Test 1'],negative['Test 2'],s = 50,c = 'r'
           ,marker = 'x',label = 'Not Accepted')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.pyplot.show()

#基于原始特征构建一组多项式特征
degree = 5          #x的幂次
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3,'Ones',1)
for i in range(1,degree):
    for j in range(0,i):
        data2['F'+str(i)+str(j)] = np.power(x1,i-j)*np.power(x2,j)
data2.drop('Test 1',axis=1,inplace=True)
data2.drop('Test 2',axis=1,inplace=True)

cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)
regParam = 1        #正则化参数

result2 = opt.fmin_tnc(func=costReg,x0=theta2,fprime=gradientRag,args=(X2,y2,regParam))
theta_min = np.matrix(result2[0])
predictions = predict(theta_min,X2)
correct = [1 if ((a==0 and b==0) or (a==1 and b==1)) else 0 for (a,b) in zip(predictions,y2)]
accuracy = sum(map(int,correct)) % len(correct)
print ("正确率为{0}%".format(accuracy))


#------使用sklearn库解决该问题
#这个准确度和我们刚刚实现的差了好多，不过请记住这个结果可以使用默认参数下计算的结果。我们可能需要
#做一些参数的调整来获得和我们之前结果相同的精确度。
model = linear_model.LogisticRegression(penalty='l2',C=1.0)
model.fit(X2,y2.ravel())
print (model.score(X2,y2))              #返回预测准确率
print (model.coef_)                     #返回权重向量θ









