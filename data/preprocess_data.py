#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import indictor
import data_const
import pandas as pd
from imp import reload 
from data import util
reload(data_const)
""" 
  预处理数据data，将需要的指标计算填充，并返回填充后数据
""" 
def preprocess_data(data_copy,basic_env=None):
    data_copy.dropna(inplace=True)#相信这一步数据没有 inf
    if len(data_copy)<2:
        print 'new stock'
        data_copy.loc[1]=data_copy.iloc[0]
    if False:
        time_coefficient = util.time_coefficient( str(data_copy.iloc[-1]['date']) )
        ll=pd.Series(data_copy.iloc[-1])
        ll['volume'] = data_copy.loc[data_copy.index[-1]]['volume'].item() + (1-time_coefficient)*data_copy.loc[data_copy.index[-2]]['volume'].item()
        data_copy.loc[data_copy.index[-1]]=ll
    columns = [str(data_copy.index.name)]+[str(u_date) for u_date in data_copy.columns]
    if set(data_const.Meta_Index_Needing).issubset(set(columns)):
        if set(data_const.Feature).issubset(set(columns)):
            return data_copy
        else:
            for item in data_const.Feature:
                columns = [str(data_copy.index.name)]+[str(u_date) for u_date in data_copy.columns]
                if item not in  columns:
                    data_copy = compute_item(data_copy,item,basic_env)
        return data_copy#[data_const.Feature]
    else:
        print 'meta data lacking'
        raise
   
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
    import pandas as pd;import tushare as ts;code='600776'
    basic_env = pd.read_csv(data_const.My_Store_Dir+'basic.csv',index_col='code');
    code_basic = basic_env.loc[int(code)];
    try:
        #codata = pd.read_csv(data_const.My_Database_Dir+code+data_const.Suffix)
        raise
    except Exception ,e:
        codata = ts.get_k_data(code); #已经没有复权了ei数据文档也有些不对
    codata = preprocess_data( codata,basic_env=code_basic );
    columns = ['p_change','volume','PVT'];n_want=6;
    print codata.head(n_want)[columns],'\n',codata.tail(n_want)[columns]
    codata.to_csv(data_const.My_Database_Dir+code+data_const.Suffix,index=False)
    codata.to_csv(data_const.Project_Root_Dir+'/resources/codata.csv');tail_n=150
    if True:
        import matplotlib.pyplot as plt
        #plt.scatter(codata['review_pc'], codata['PVT'])
        plt.figure(figsize=(75,55))
        plt.subplot(221)#括号里的（mnp）：m表示是图排成m行，n表示图排成n列，p表示位置
        plt.plot(codata[['PVT']].tail(tail_n));plt.grid()  # 生成网格
        print codata[['VT']]
        plt.subplot(222);
        plt.plot(codata[['VT']].tail(tail_n));plt.grid;
        plt.subplot(223)
        plt.plot(codata['close'].tail(tail_n));plt.grid()  # 生成网格
        plt.show();
if __name__ == '__main__':
    test()
    
    
