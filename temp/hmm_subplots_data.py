#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import tushare as ts
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from hmmlearn.hmm import GaussianHMM
quotes = ts.get_k_data('601988')
date = np.array(quotes['date'])
close = np.array(quotes['close'])
volume=np.array(quotes['volume'])
x=np.column_stack([close,volume])
#print(open,'open_data_over')
#以下是用HMM训练 运行高斯HMM
print("fitting to HMM and decoding ...", end="")
#创建一个HMM实例并执行fitting
model = GaussianHMM(n_components=5,covariance_type="diag",n_iter=1000).fit(x)
#预测内部隐藏状态的最佳顺序
hidden_states=model.predict(x)
#以下是画图
print("Transition matrix")
print(model.transmat_)
print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ",model.means_[i])
    print("val = ",np.diag(model.covars_[i])) 
#plt.subplots  有s和没有s  有差别的
fig , axs = plt.subplots(model.n_components,sharex=True,sharey=True)
colormap=cm#解决傻逼报错下面
#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#在指定的间隔内返回均匀间隔的数字
colours = colormap.rainbow(np.linspace(0,1,model.n_components))
for i ,(ax,colour) in enumerate(zip(axs,colours)):
    #使用花哨索引来绘制每个状态的数据
    mask = hidden_states == i
    #print(i,'state',mask)
    ax.plot_date(date[mask],close[mask],".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))
    ax.grid(True)
plt.show()