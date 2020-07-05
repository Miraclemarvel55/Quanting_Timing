#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
def getDataAfterPCA(codata):
    pca=PCA(n_components='mle') #n_components=2 指定想要的主成分个数 mle 自动指定个数
    principal_components=pca.fit_transform(codata) #训练pca模型之后返回训练的数据以及模型，fit（）单独训练模型只返回模型
    #print principal_components
    principal_components = pd.DataFrame(principal_components)
    #print type(principal_components)
    return principal_components,pca
if __name__ == '__main__':
    data=load_iris()
    y=data.target
    x=data.data
    reduced_x,pca=getDataAfterPCA(x)
    print type(reduced_x),reduced_x.size,reduced_x.shape
    print '信息比例',pca.explained_variance_ratio_,'sum=',(sum(pca.explained_variance_ratio_))
    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    green_x,green_y=[],[]
     
    for i in range(len(reduced_x)):
        if y[i] ==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')
    #plt.show()



