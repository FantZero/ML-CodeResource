# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 18:28:41 2018
1-数据可视化
@author: jz
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex7data1.mat')
data1 = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
sns.set(context='notebook',style='white')
sns.lmplot('X1','X2',data=data1,fit_reg=False)
plt.show()