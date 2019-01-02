# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 19:30:56 2018

@author: jz
"""
import numpy as np
import tkinter as tk
import regTree
import matplotlib
matplotlib.use('TkAgg')     #将matplotlib后端设置为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#以用户输入的终止条件为参数绘图
def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree = regTree.createTree(reDraw.rawDat,regTree.modelLeaf,regTree.modelErr,(tolS,tolN))
        yHat = regTree.createForeTree(myTree,reDraw.testDat,regTree.modelTreeEval)
    else:
        myTree = regTree.createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = regTree.createForeTree(myTree,reDraw.testDat)
    #绘制真实值
    reDraw.a.scatter(reDraw.rawDat[:,0].tolist(),reDraw.rawDat[:,1].tolist(),s=5)
    #绘制预测值
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)
    reDraw.canvas.show()

#从文本输入框中获取树创建终止条件
def getInputs():
    try:tolN = int(tolNentry.get())
    except:
        tolN = 10
        print ('enter Integer for tolN')
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print ('enter Float for tolS')
        tolSentry.delete(0,END)
        tolSentry.insert(0,'10')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolS,tolN)


root = tk.Tk()

reDraw.f = Figure(figsize=(5,4),dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

#标签部件
tk.Label(root,text='Plot Place Holder').grid(row=0, columnspan=3)
#文本框输入部件
tk.Label(root,text='tolN').grid(row=1,column=0)
tolNentry = tk.Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')
tk.Label(root,text='tolS').grid(row=2,column=0)
tolSentry = tk.Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')
#按钮部件
tk.Button(root,text='ReDraw',command=drawNewTree).grid(row=1,column=2,rowspan=3)
#复选按钮部件
chkBtnVar = tk.IntVar()
chkBtn = tk.Checkbutton(root,text='Model Tree',variable=chkBtnVar)
chkBtn.grid(row=3,column=0,columnspan=2)
reDraw.rawDat = np.mat(regTree.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)

reDraw(0.1,10)
root.mainloop()


















