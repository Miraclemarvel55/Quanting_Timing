'''
Created on 2018年11月9日

@author: feiyu
'''
from ml_stock import ml_const,ml_all
import pandas as pd
import data.data_const as dtcst
import data.util as util
import matplotlib
matplotlib.use("TKAgg")
#matplotlib.use("Pdf")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
def advice_func_general(temp):
    return temp
def simu_show_main(ml_func=ml_all.naive_bayes,advice_func=advice_func_general,review_num=45,review_end=1):
    codes=dtcst.My_Wanted_codes
    if dtcst.RunningOnServer:
            import random
            codes = random.sample(codes, len(codes)/10)
    len_codes = len(codes);i=1
    for code in codes:
        #code='603045'
        print i,'/',len_codes,'simulation',code;i+=1;
        try:
            codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
        except IOError,e:
            print e;i-=1;len_codes-=1
            continue
        if len(codata)<=review_num+55:
            print 'data amount too small'
            i-=1;len_codes-=1
            continue
        simu_show(codata, ml_func, advice_func, review_num, review_end)
        
def simu_show(data,ml_func,advice_func,review_num=45,review_end=1):
    data['test'] =0;data.reset_index(drop=True);len_=len(data)
    for i in range(review_end,review_num)[::-1]:
        index = len_-i#注意head()和data.loc() 参数含义的不同head参数表示前第几行，loc参数表示索引
        simulation_data = data.head(index)
        #训练函数必须已经内置去除第一条数据的处理，否则这里要自己手动处理
        ml_func_result = ml_func(simulation_data)
        #print 'ml_func_result',ml_func_result
        advice = advice_func(ml_func_result)
        close = simulation_data.tail(1)['close'].item()
        if advice==True:
            advice=close
        else:advice=close/2
        idx = index-1
        data.loc[idx,'test']=advice
        print simulation_data.tail(1)['date'].item(),close ,advice
    print data
    data = data.set_index('date')
    data = data.tail(review_num-1)[['close','ema5','MACD','test']]
    print data
    data.plot();plt.grid(True);plt.show()