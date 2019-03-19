#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#导入相关的模块
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import data.data_const as data_const
import tushare as ts
sns.set_style('white')
# 获取数据
data = ts.get_k_data('601988')
code = '601988'
data = pd.read_csv(data_const.Test_Database_Dir+code+data_const.Suffix)
data1 = data['high']
data2 = data['low']
tradeVal = data1 + data2 #1融资融券数据总和
tradeVal = pd.DataFrame(tradeVal)
tradeDate = pd.to_datetime(data['date'][5:])#日期列表
volume = data['volume'][5:]#2 成交量数据
volume = np.array(volume)
closeIndex = data['close'] # 3 收盘价数据
deltaIndex = np.log(np.array(data['high'])) - np.log(np.array(data['low'])) #3 当日对数高低价差
deltaIndex = deltaIndex[5:]
logReturn1 = np.array(np.diff(np.log(closeIndex))) #4 对数收益率
logReturn1 = logReturn1[4:]
logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))# 5日 对数收益差
logReturnFst = np.array(np.diff(np.log(tradeVal[tradeVal.columns[0]])))[4:]
closeIndex = closeIndex[5:]

print type(logReturn1),type(logReturn5),type(deltaIndex),type(volume),type(logReturnFst)
print logReturn1.size,logReturn5.size,deltaIndex.size,volume.size,logReturnFst.size
X = np.column_stack([logReturn1,logReturn5,deltaIndex,volume,logReturnFst]) # 将几个array合成一个2Darray
print X.size,X.shape
# Make an HMM instance and execute fit
states_num = 4
model = GaussianHMM(n_components=states_num, covariance_type="diag", n_iter=1000).fit(X)
# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)
#图形显示
plt.figure(figsize=(15, 8)) 
for i in range(model.n_components):#model.n_components=6代表6个不同的隐藏状态用i遍历6个状态
    idx = (hidden_states==i) #如果hidden_states==i则返回true 否者返回false
    print type(idx),idx.size,tradeDate[idx].size,closeIndex[idx].size
    plt.plot_date(tradeDate[idx],closeIndex[idx],label='%dth hidden state'%i,lw=1) 
    #plt.plot_date画时间，当hidden_states==i 那天为ture时讲收盘价画上去组成一条线总共有6跳线进行叠加第一个参数时间，
    #后面的Y轴数值，最新日期，收盘指数  画的状态线是'.'点状 %d 第%i种状态，lw=1 线粗=1
plt.legend()    #显示图例，就是右上角的标签
plt.grid()     #显示网格    
plt.show()

    