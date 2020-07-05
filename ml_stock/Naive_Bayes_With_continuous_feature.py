#!/usr/bin/env python
#coding:utf-8
#朴素贝叶斯算法
def p_change_to_class(pc):
    #四舍五入法连续变离散映射
    try:
        return int(pc+(0.5 if pc>0 else -0.5) )
    except:
        print('error p_change_to_class',pc);return 0
def getTrainSet(codata):
    naive_bayes_needing_attributes = ['turnover_rate','p_change','amplitude','RSV','high_d','low_d','ema5_d','ema5_d_d','review_pc']
    df = codata[naive_bayes_needing_attributes].reset_index(drop=True) 
    for idx in range(len(df)):
        df.loc[idx,'review_pc'] = p_change_to_class(df.loc[idx,'review_pc'])
    import numpy as np;dataSetNP = np.array(df)  #将数据由dataframe类型转换为数组类型
    trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]   #训练数据x1,x2
    labels = dataSetNP[:,dataSetNP.shape[1]-1]        #训练数据所对应的所属类型Y
    return trainData, labels
'''if not predict_features:
        print weight_list
        trainData = np.repeat(trainData,weight_list,axis=0)
        labels = np.repeat(labels,weight_list,axis=0)'''
def weight_generater(d,d_all,review_pc,k=1,trainData=None,review_pcs=[]):
    import math,numpy #review_pc:股价的明日变化值 d_now:总的天数 w2:0--d-1-->0--pi
    #d_all=len(review_pcs);   #wt_list=[];k调节大涨大跌与近期时间加权的系数默认为1
    w1_func = lambda review_pc:numpy.e**(abs( review_pc )/2) #1到148倍
    w2_func =lambda d,d_all_1=(d_all-1)*1.0000:(-math.cos( d/d_all_1*math.pi )+2)*50 #50到150倍
    K=148;P0=K/2;shift=d_all/2;r=0.05;#P0:初值,k:终值,r:变化控制参数
    #y=148*74*e^(0.05*(x-100))/(148+74*(e^(0.05*(x-100))-1))+1
    w3_func = lambda t :K*P0*math.e**(r*(t-shift))/( K+P0*(math.e**(r*(t-shift))-1) )+1
    max_times=148;end=d_all+0.0000;
    #y=148/( -(x-201) )+1
    w4_func = lambda x:max_times/( -(x-end) )+1
    w5_func = lambda d : 148-3*(d_all-d) if d>=d_all-15 else 1
    w6_func = lambda d : 148 if d==d_all-1 else 1
    return int( w1_func(review_pc)+k*w4_func(d) +0.5 ) #有需要可以+0.5 取四舍五入
    
def classify(trainData, labels, features):
    #print 'entering classify',type(trainData),trainData.shape,trainData.size,labels,len(labels),features
    if len(labels)==0:
        print 'NVBYS-classify traindata is 0 line'
        raise
    #求labels中每个label的先验概率
    labels = list(labels)    #转换为list类型
    P_y = {}       #存入label的概率
    for label in labels:
        P_y[label] = labels.count(label)/float(len(labels))   # p = count(y) / count(Y)
    '''
    #求label与feature同时发生的概率 适合离散特征
    P_xy = {}
    for y in P_y.keys():
        y_index = [i for i, label in enumerate(labels) if label == y]  # labels中出现y值的所有数值的下标索引
        for j in range(len(features)):      # features[0] 在trainData[:,0]中出现的值的所有下标索引
            x_index = [i for i, feature in enumerate(trainData[:,j])if feature == features[j]]
            xy_count = len(set(x_index) & set(y_index))   # set(x_index)&set(y_index)列出两个表相同的元素
            pkey = str(features[j]) + '*' + str(y)
            P_xy[pkey] = xy_count / float(len(labels))'''
    #求P[X/Y] 对同一类的特征的连续数据进行建模求概率密度，似然概率
    P = {}
    for y in P_y.keys():
        y_index = [i for i, label in enumerate(labels) if label == y]  # labels中出现y值的所有数值的下标索引
        for j in range(len(features)):      # features[0] 在trainData[:,0]中出现的值的所有下标索引
            #traindata_of_x_in_y= list(set(trainData[y_index,j].tolist()))#ｓｅｔ化除去重复数据，但是只要除去０　就好这里，认为其它数据值重复可能性较小
            traindata_of_x_in_y= trainData[y_index,j].tolist()
            #print traindata_of_x_in_y
            import statsmodels.api as sm
            showingneeding=False
            if showingneeding:
                import  matplotlib as mpl
                mpl.use('Agg')
                import matplotlib.pyplot as plt  
                import seaborn as sns
                import matplotlib.pyplot as plt2
                print 'plt后端',plt.get_backend()
                print type(traindata_of_x_in_y),'len:',len(traindata_of_x_in_y),traindata_of_x_in_y
                print '即将显示此类此特征数据分布'
                sns.distplot(traindata_of_x_in_y,rug=True)#概率密度图像 rug：控制是否生成观测数值的小细条，通过hist和kde参数调节是否显示直方图及核密度估计(默认hist,kde均为True)
                sns.kdeplot(traindata_of_x_in_y,cumulative=True)#纯累计概率函数曲线
                #上面两个sns 图像联合使用可以得到概率密度和概率累积函数共x轴图
                plt.xlabel('feature:'+str(j));plt.ylabel('pdf');
                plt.title('class:'+str(y)+' feature:'+str(j)+'--pdf')
                # 概率密度
                dens = sm.nonparametric.KDEUnivariate(traindata_of_x_in_y)
                dens.fit()
                print features[j],dens.evaluate(features[j]),'参考点0:',dens.evaluate(0)
                plt.show()
                '''plt2.plot(dens.cdf)#cdf累计概率
                plt2.show()'''
            #print 'traindata_of_x_in_y',traindata_of_x_in_y
            dens = sm.nonparametric.KDEUnivariate(traindata_of_x_in_y)
            try:
                dens.fit();
                wanted_x_density = dens.evaluate(features[j])
            except:
                print('NVBYS.classify.dens.fit','class:',y,'feature:',j,'traindata_of_x_in_y',traindata_of_x_in_y)
                wanted_x_density=0
            '''旧版手动高斯计算的概率密度
            import data_fit as dtft;import numpy as np;
            narray=np.array(traindata_of_x_in_y)
            sum1=narray.sum()
            narray2=narray*narray
            sum2=narray2.sum()
            N = len(traindata_of_x_in_y)
            mean=sum1/N
            var=sum2/N-mean**2
            if var==0:
                var = 0.1#只有一行数据，var =0，为避免var出现零，但是将高斯分布，逼成很窄的曲线
            #print mean,var,N,y_index,traindata_of_x_in_y
            wanted_x_density = dtft.f_gauss(features[j], 1, mean, 0,var)'''
            pkey = str(features[j]) + '|' + str(y)
            P[pkey] = wanted_x_density
    #求features[2,'S']所属类别
    F = {}   #[2,'S']属于各个类别的概率
    for y in P_y.keys():
        F[y]=P_y[y]# 先验概率
        for x in features:
            F[y] = F[y]*P[str(x)+'|'+str(y)]     #P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
    #features_label = max(F, key=F.get)  #概率最大值对应的类别
    #不选用概率最大那个，可以选用期望值
    features_label='-0';Z=0#归一化系数Z
    expection = 0; import math
    for key,value in F.iteritems():
        if math.isinf(value) or math.isnan(value):continue #避免只有一个数据的选择奇怪哈
        else :
            Z+=value;  expection+=key*value
    if Z>0:expection/=Z;features_label=str(expection-0.001)#-0.001避免极小的正值
    return features_label

if __name__ == '__main__':
    import pandas as pd
    import data.data_const as dtcst
    from data.preprocess_data import preprocess_data
    codes=dtcst.My_Wanted_codes
    codes=['603032']
    #print dtcst.My_Database_Dir
    for code in codes:
        print code
        data = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix);import data.preprocess_data as pre_process
        codata = preprocess_data(data, basic_env=None);
        #Drop rows by index
        castrate_data = codata.drop([codata.index[0], codata.index[len(codata)-1]])
        trainData, labels = getTrainSet(castrate_data)
        features,_ = getTrainSet(codata.tail(1))
        #print '特征向量',type(features),features
        # 该特征应属于哪一类
        result = classify(trainData, labels, features[0])
        print '特征',features,'属于',result
        
        
        