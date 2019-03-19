#!/usr/bin/env python
#coding:utf-8
# 极大似然估计  朴素贝叶斯算法
import pandas as pd
import numpy as np
import data.data_const as dtcst

def p_change_to_class(p_change):
    #print 'entering p_change_to_class:',p_change
    pc=p_change;pclass = 'ultra'
    if 0<=pc<1:
        pclass = 'A'
    elif 1<=pc<2:
        pclass = 'B'
    elif  2<=pc<3 :
        pclass = 'C'
    elif  3<=pc<5.5 :
        pclass = 'D'
    elif  5.5<=pc :
        pclass = 'E'
    elif  -3<=pc<0 :
        pclass = '-A'
    elif  -6<=pc<-3 :
        pclass = '-B'
    elif  pc<-6 :
        pclass = '-E'
    #print 'leaving p_change_to_class:',pclass
    return pclass
def getTrainSet(codata):
    wanted_data = codata[['turnover_rate','MACD','MACD_d','DIFF_d','K_d','p_change']].reset_index(drop=True) 
    for idx in range(len(wanted_data)):
        wanted_data.loc[idx,'p_change'] = p_change_to_class(wanted_data.loc[idx,'p_change'])
    dataSetNP = np.array(wanted_data)  #将数据由dataframe类型转换为数组类型
    trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]   #训练数据x1,x2
    labels = dataSetNP[:,dataSetNP.shape[1]-1]        #训练数据所对应的所属类型Y
    if(len(codata)==1):trainData=trainData[0]#此为欲分类数据不是训练数据，只需要一维
    return trainData, labels

def classify(trainData, labels, features):
    #求labels中每个label的先验概率
    labels = list(labels)    #转换为list类型
    P_y = {}       #存入label的概率
    for label in labels:
        P_y[label] = labels.count(label)/float(len(labels))   # p = count(y) / count(Y)

    #求label与feature同时发生的概率
    P_xy = {}
    for y in P_y.keys():
        y_index = [i for i, label in enumerate(labels) if label == y]  # labels中出现y值的所有数值的下标索引
        for j in range(len(features)):      # features[0] 在trainData[:,0]中出现的值的所有下标索引
            x_index = [i for i, feature in enumerate(trainData[:,j])if feature == features[j]]
            xy_count = len(set(x_index) & set(y_index))   # set(x_index)&set(y_index)列出两个表相同的元素
            pkey = str(features[j]) + '*' + str(y)
            P_xy[pkey] = xy_count / float(len(labels))

    #求条件概率
    P = {}
    for y in P_y.keys():
        for x in features:
            pkey = str(x) + '|' + str(y)
            P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])    #P[X1/Y] = P[X1Y]/P[Y]

    #求features[2,'S']所属类别
    F = {}   #[2,'S']属于各个类别的概率
    for y in P_y:
        F[y] = P_y[y]
        for x in features:
            F[y] = F[y]*P[str(x)+'|'+str(y)]     #P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]

    features_label = max(F, key=F.get)  #概率最大值对应的类别
    return features_label
'''
if __name__ == '__main__':
    for code in dtcst.Test_Wanted_codes:
        #code = '000002'
        codata = pd.read_csv(dtcst.Test_Database_Dir+code+dtcst.Suffix)
        # 训练数据,标签
        trainData, labels = getTrainSet(codata)
        features,_ = getTrainSet(codata.tail(1))
        print codata.tail(1)
        # 该特征应属于哪一类
        result = classify(trainData, labels, features)
        print features,'属于',result
'''