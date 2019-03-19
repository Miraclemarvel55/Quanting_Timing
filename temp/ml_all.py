#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import data.data_const as dtcst
import ml_const as mlcst
import Naive_Bayes_With_continuous_feature as nv_bys
import hmm_predict as hmm_p
import ml_svm as svm
import numpy as np
import data.util as util

def ml_all(realtime_data = pd.read_csv(dtcst.My_Store_Dir+'realtime_data.csv').set_index('code')):
    predict_table = pd.DataFrame(columns=mlcst.attributes)
    basic_env = pd.read_csv(dtcst.My_Store_Dir+'basic.csv',index_col='code')
    predict_table['Code_Index'] = dtcst.My_Wanted_codes
    predict_table = predict_table.set_index("Code_Index")
    i=1;codes = dtcst.My_Wanted_codes;print codes
    for code in codes:
        print '\n',i,'/',len(codes),'ml_stock',code;i+=1;
        code_rt_dt = realtime_data.loc[int(code)]
        code_basic = basic_env.loc[int(code)]
        if code_rt_dt['high'].item()==code_rt_dt['low'].item() and code_rt_dt['open'].item()==0 :
            continue
        try:
            codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
        except IOError:
            continue
        codata.dropna(inplace=True)
        if len(codata)<=15:
            print 'data amount too small'
            continue
        temp = []
        temp.append(code_rt_dt['name'])
        temp.append(default_fun(codata))
        temp .append(quantity_potiential_judge(codata))
        temp.append(hmm_state(codata))
        temp .append(naive_bayes(codata))
        temp .append( macd_diff_judge(codata))
        temp.append(additional_judge(code_basic))
        temp.append(getCoefficient_of_Variation(codata))
        temp.append(get_macd_d_num_in5days(codata))
        predict_table.loc[code] = temp
    predict_table.dropna(inplace=True)
    print predict_table
    print dtcst.My_Store_Dir
    predict_table.to_csv(dtcst.My_Store_Dir+'predict_table.csv')
    recommend_table = recommend_table_generator(predict_table)
    recommend_table.to_csv(dtcst.My_Store_Dir+'recommend_table.csv')
    print recommend_table
    
#使用macd一阶导数以及二阶导数获得最块增长点的机会
#某些股票在MACD=0和其他一些临界点会反转，此处使用0反转
def macd_diff_judge(codata):
    last_data = codata.tail(1)   
    return (last_data['MACD'].item()>0 or last_data['MACD_d_d'].item()>0 or last_data['DIFF_d'].item()>0)\
             and last_data['MACD_d'].item()>0

#默认使用五日平滑后的曲线斜率大于零策略
def default_fun(codata):
    turnover_rate_thresh = codata.tail(30)['turnover_rate'].mean()
    last_data = codata.tail(1)
    #高点越高低点也越高，体现短期买入量能大
    return (last_data['high_d'].item()>0 and last_data['low_d'].item()>0)\
         and (last_data['turnover_rate'].item()>turnover_rate_thresh or \
         last_data['ema5_d_d'].item()>0) and last_data['ema5_d'].item()>0
def kdj_ema_quick_d(codata):
    last_data = codata.tail(1)
    return  last_data['D_d'].item()>0 and last_data['K_d'].item()>0
#隐马尔可夫筛选
def hmm_state(codata):
    #print 'entering hmm_state(codata)',
    codata=codata.tail(65)
    if len(codata)<25:return False
    state,_ = hmm_p.get_last_state(codata)
    return state
def ml_svm_predict(codata):
    codata=codata.tail(65)
    #print 'entering ml_svm_predict',
    if len(codata)<25:return False
    return svm.SVM_SVR_predict(codata)
#贝叶斯筛选
def naive_bayes(codata):
    codata=codata.tail(65)
    #print 'entering naive_bayes',
    if len(codata)<25:return False
    # 训练数据,标签
    #Drop rows by index
    castrate_data = codata.drop([codata.index[0], codata.index[len(codata)-1]])
    trainData, labels = nv_bys.getTrainSet(castrate_data)
    features,_ = nv_bys.getTrainSet(codata.tail(1))
    #print '-',type(features),features
    # 该特征应属于哪一类
    result = nv_bys.classify(trainData, labels, features)
    #print features,'属于',result  
    return  not '-' in result

#个人经验限制self_experiences_judge
def quantity_potiential_judge(codata):#量能判断
    lastdata = codata.tail(1);
    return lastdata['turnover_rate'].item()>mlcst.Turn_Over_rate_thresh and lastdata['amount'].item()>mlcst.Amount_thresh\
            and getlast7days_p_change(codata)<mlcst.last7days_p_change_thresh and  lastdata['nmc'].item()>mlcst.nmc_thresh
            #流通市值判断
def env_relative_judge(codata):
    lastdata = codata.tail(1);
    lastdate = lastdata['date']
    MACD_d = dtcst.env_sh_data.loc[lastdate]['MACD_d'].item()
    if MACD_d>=0:return True
    else: return False
#东北企业筛选限制
def isnortheast(code_basic):
    if not code_basic['area']== '黑龙江' and not code_basic['area']== '吉林' and not code_basic['area']== '辽宁':
        return True
    else:return False
def additional_judge(codata,code_basic):
    return isnortheast(code_basic)
#变异系数 -- 描述股价波动情况,计算类似方差,但是可以除以均值,使得各个股票可以在基数不一样情况下,比较波动
def getCoefficient_of_Variation(codata):
    close = codata['close']
    narray=np.array(close)
    sum1=narray.sum()
    narray2=narray*narray
    sum2=narray2.sum()
    N = len(close)
    mean=sum1/N
    var=sum2/N-mean**2
    if var==0:
        var = mean/100+0.1
    Coefficient_of_Variation = var/mean/mean
    return Coefficient_of_Variation
#近7日涨跌幅度
def getlast7days_p_change(codata):
    last7_close = codata.tail(7)['close']
    last7days_p_change = (last7_close.tail(1).item()-last7_close.head(1).item())/last7_close.head(1).item()*100
    return abs(last7days_p_change)
#macd天数和上市天数的积 选择偏向次新和macd少的
def get_macd_d_num_in5days(codata):
    num = 0
    last5data = codata.tail(5)
    for index,row in last5data.iterrows():
        if row['MACD_d']>0:num+=1
    
    return num*len(codata)

def recommend_table_generator(predict_table):
    recommend_table = predict_table.drop(predict_table.index)#生成一个只有列名的空dataframe
    code_func_map = util.code_func_map()
    for Code_Index,item in predict_table.iterrows():
        try:
            test_methods = code_func_map[Code_Index]
        except:
            test_methods = ['default']
        temp = [item[test_method] for test_method in test_methods]
        print Code_Index,test_methods,temp
        if not False in temp:
            recommend_table.loc[Code_Index] = item
    df = recommend_table
    print recommend_table
    print 'entering recommend_table_generator pre recommend_table len: ',len(recommend_table)
    mean__Coefficient_of_Variation = predict_table['Coefficient_of_Variation'].mean()
    #单独构造 series 原因是： Naive_Bayes 的比较不能使用相应比较的==或者!=符号
    #Naive_Bayes_select = pd.Series([not item.find('-')>-1for item in df['Naive_Bayes']], index=df.index.tolist())
    #shorterm_fitting_select = pd.Series([abs(item)<mlcst.shorterm_fitting_thresh for item in df['Shorterm_Fitting']], index=df.index.tolist())
    #A common operation is the use of boolean vectors to filter the data. 
    #The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses--().
    recommend_table = df[(  (df['self_experiences_judge']==True) & (df['additional_judge']==True) & (df['in_market_days_num']<912) )]
    #偏好次新股,上市天数较少的排列在前 & (df['Coefficient_of_Variation']>=mean__Coefficient_of_Variation)
    recommend_table.sort_values(by="in_market_days_num" , ascending=True,inplace=True)
    return recommend_table

if __name__ == '__main__':
    print '模块测试'
    ml_all()