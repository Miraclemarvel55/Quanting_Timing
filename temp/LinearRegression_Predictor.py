#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from data import util
from data import data_const
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    code = '000002'
    df = pd.DataFrame(pd.read_csv(data_const.Project_Root_Dir+data_const.Test_Database_Dir+code+data_const.Suffix))
    df.dropna(inplace=True)
    X = df[data_const.Feature]
    y = df["p_change"]
    y= y.shift(-1) #预测标签前置
    #删除特征值首行：很多0值，删除预测标签尾值：没有预测标签
    #综合需要删除首尾两行
    X.drop([0, len(X)-1],inplace=True)
    del y[y.index[0]];del y[y.index[len(y)-1]]
    X = preprocessing.scale(X)
    X_train, X_test, y_train ,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
    #clf = svm.SVR()
    clf=LinearRegression(n_jobs=-1) #n_jobs=-1代表用尽可能的线程来执行
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    forecast_set = clf.predict(X_test).tolist()
    #forecast_set = pd.DataFrame({'y_predict':forecast_set},columns =['y_predict'])
    y_test = y_test.tolist()
    compare_y_dic = dict(zip(y_test,forecast_set))
    print len(y_test)
    error =0
    for key,value in compare_y_dic.items():  
        #print('{key}:{value}'.format(key = key, value = value))  
        key = float(key); value = float(value)
        if key*value < 0:
            error +=1
            print key - value
    print '错误率: ',float(error)/len(y_test)    
        
        