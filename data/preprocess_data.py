#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import indictor
import data_const
from imp import reload 
reload(data_const)
""" 
  预处理数据data，将需要的指标计算填充，并返回填充后数据
""" 
def preprocess_data(data_copy,basic_env=None):
    data_copy.dropna(inplace=True)#相信这一步数据没有 inf
    if len(data_copy)<2:raise
    columns = [str(data_copy.index.name)]+[str(u_date) for u_date in data_copy.columns]
    if set(data_const.Meta_Index_Needing).issubset(set(columns)):
        if set(data_const.Feature).issubset(set(columns)):
            return data_copy
        else:
            for item in data_const.Feature:
                columns = [str(data_copy.index.name)]+[str(u_date) for u_date in data_copy.columns]
                if item not in  columns:
                    data_copy = compute_item(data_copy,item,basic_env)
        return data_copy[data_const.Feature]
        #return data_copy
    else:
        print 'meta data lacking'
        raise;return False
   
def compute_item(data,item,basic_env=None):
    if data_const.Derivative_str in str(item[-3:]):
        try:
            return indictor.add_index_derivative(data,item[:-2])#等价indictor.add_index_derivative(data,item.strip(data_const.Derivative_str))
        except:
            data = compute_item(data, item[:-2], basic_env)
            return compute_item(data, item, basic_env)
    elif basic_env is None:
        return indictor.add_wanted_index(data,item)
    else:
        return indictor.add_wanted_index(data,item,basic_env)
def test():
    import pandas as pd;import tushare as ts;code='000001';today='2017-02-11';
    try:
        codata = pd.read_csv(data_const.My_Database_Dir+code+data_const.Suffix)
        print codata.iloc[-250]['date']
        #codata = ts.get_k_data(code,today)
        basic_env = pd.read_csv(data_const.My_Store_Dir+'basic.csv',index_col='code')
        code_basic = basic_env.loc[int(code)]
        print 'leaving try'
    except Exception ,e:
        print e
        code_jq=code+'.XSHG'
        from jqdata import *;
        try:
            from jqdatasdk import *
        except:
            pass
        q=query(valuation.code,valuation.circulating_cap).filter(valuation.code.in_([code_jq]))
        basic_env = get_fundamentals(q,date=today);
        basic_env = basic_env.set_index('code');basic_env.columns = ['outstanding'];code_basic=basic_env.loc[code_jq]
        end_date=today;count = 6;
        codata = get_price(code_jq,end_date=end_date, frequency='daily', fields=None, skip_paused=True, fq='pre',count=count)
        print 'before preprocess',codata.head(2),codata.tail(2)
        #codata = ts.get_k_data(code,'2018-01-01')
        print e,'preprocess_data.test error'
    print '模块测试';
    codata = preprocess_data( codata,basic_env=code_basic )
    columns = ['p_change','volume','PVT'];n_want=6
    print codata.head(n_want)[columns],'\n',codata.tail(n_want)[columns]
    codata.to_csv('./codata.csv')
    if True:
        import matplotlib.pyplot as plt
        #plt.scatter(codata['review_pc'], codata['PVT'])
        plt.figure(figsize=(75,55))
        plt.subplot(211)#括号里的（mnp）：m表示是图排成m行，n表示图排成n列，p表示位置
        plt.plot(codata[['close']].tail(55).head(30));plt.grid()  # 生成网格
        plt.subplot(212)
        plt.plot(codata['PVT'].tail(55).head(30));plt.grid()  # 生成网格
        plt.show()
if __name__ == '__main__':
    test()
    
    
