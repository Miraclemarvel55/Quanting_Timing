#!/usr/bin/env python
# -*- coding:utf-8 -*- 
'''
数据降维, 主成分分析PCA法和相关系数法.网上能公开获取的数据越来越多,质量也越来越高,其维度也越来越高.
高维度的数据会导致计算量和计算复杂度加大.特别是有些维度之间是强相关的,这些冗余数据对算法有害无益.
所以数据降维是数据准备工作中不可或缺的步骤.
PCA的原理网上有很多高质量的讲解.这里仅以调用scikit-learn中成熟库来举例,
对Tushare提供的企业盈利能力数据进行处理
'''
import tushare as ts

import sklearn.decomposition as skd

#profit = ts.get_profit_data(2017,2) 
profit = ts.get_k_data('601988')
#print profit.head
profit.drop('date', axis=1, inplace=True) 
print '显然数据中的公司代码和公司名称不该参加计算'

profit.drop('code', axis=1, inplace=True)
print profit
profit.dropna(inplace=True) 
print '数据中的空值需要事先清除'

pca = skd.PCA(n_components='mle')  
print '目标维数可以让算法自己确定,故先设为mle'

newprofit = pca.fit_transform(profit)  
print '把降维结果输出到新的变量里. 这里有两个概念需要理解清楚,1是用profit数据对pca这个类进行降维训练,2是用pca对profit进行降维.'

print pca.explained_variance_ratio_.sum()
print '显示新维度的贡献率.本例中前三维的贡献率就达到99%了'
print newprofit