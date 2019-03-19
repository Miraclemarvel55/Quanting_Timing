#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.svm import SVC,SVR #SVC(support vectors classification):分类， SVR(support vectors regression)：回归
import MyPCA as PCA
#外部调用函数
def SVM_SVR_predict(codata):
    df = codata[["p_change","amplitude",'MACD_d','MACD_d_d','review_pc']];
    df_drop_last_line = df.drop(df.index[-1])
    #print df,df_drop_last_line
    X = df_drop_last_line[["p_change","amplitude",'MACD_d','MACD_d_d']]
    y = df_drop_last_line["review_pc"]
    df.pop('review_pc') #除去目标列
    #X_train, X_test, y_train ,y_test = cross_validation.train_test_split(X,y,test_size=0)
    X_train=X;y_train = y;
    X_train,pca=PCA.getDataAfterPCA(X_train)
    #print X.size,X_train.size,X_test.size,type(X_test)
    kernel_type = ['rbf']
    k = kernel_type[0]
    clf_R = SVR(k)
    clf_R.fit(X_train,y_train)
    pred = clf_R.predict(pca.transform(df.tail(1)) )
    if pred[0]>0:return True
    else: return False

if __name__ == "__main__":
    import pandas as pd
    import data.data_const as dtcst
    from sklearn import cross_validation
    codes = dtcst.My_Wanted_codes
    codes = ['603598']
    num=10;train_num=155
    for code in codes:
        print code
        df = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix).tail(num+train_num)
        X = df[["p_change","amplitude",'MACD_d','MACD_d_d']]
        X,pca=PCA.getDataAfterPCA(X)
        y = df["review_pc"]
        y.index = y.index- y.index[0] #特征数据X索引已经重置
        print type(X),y.shape
        X_train, X_test, y_train ,y_test = cross_validation.train_test_split(X,y,test_size=0.2)#随机分配
        kernel_type = ['rbf'] #['linear','poly','rbf','sigmoid']
        for k in kernel_type:
            #print 'kernel_type:',k
            clf_R = SVR(k)
            clf_R.fit(X_train,y_train)
            pred = clf_R.predict(X_test.tail(num))
            print pred, '\n',y_test.tail(num)
            accuracy2 = clf_R.score(X_test,y_test)    
            #print(accuracy2)
        
        