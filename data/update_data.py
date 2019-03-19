#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tushare as ts
import pandas as pd
import data_const,time
import preprocess_data as prep_d
import indictor as idx
from util import *

#比较本地sh.csv文件是否是最近一天的是否热乎s是的话返回sh.csv的最末行日期，否则使用ts 库更新文件
def get_last_exchange_day():
    from datetime import datetime, date, timedelta
    print 'entering get_last_exchange_day'
    import sys
    reload(sys)
    sys_temp = sys
    sys_temp.setdefaultencoding('utf-8')
    date_time= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hour = int(date_time[11:13])
    minute = int(date_time[14:16])
    try:
        print 'entering try1'
        file_changetime,file_date = get_FileModifyTime(data_const.My_Store_Dir+'sh.csv')
        file_hour = int(file_changetime[11:13])
        file_minute= int(file_changetime[14:16]);time_thresh=5
        if file_date ==  getDatetimeToday().strftime("%Y-%m-%d") and (hour-file_hour)*60+minute-file_minute <time_thresh:
            data_sh_index = pd.read_csv(data_const.My_Store_Dir+'sh.csv')[data_const.Meta_Index_Needing]#避免下面重复计算
            prep_d.compute_item(data_sh_index,'MACD_d').to_csv(data_const.My_Store_Dir+'sh'+data_const.Suffix)
            last_date = data_sh_index.loc[len(data_sh_index)-1]['date']
            print 'today:',hour ,minute,'file_changetime:',file_hour,file_minute,'time_thresh:',time_thresh,'minutes'
            print 'within timethresh needing file not need change,leaving get_last_exchange_day',last_date
            return last_date
    except :
        pass
    try:
        print 'entering try 2'
        #环境数据,股票基本资料
        basic = 'basic.csv'
        ts.get_stock_basics().to_csv(data_const.My_Store_Dir+basic)
        print basic ,'ok'
        #实时行情数据
        realtime = 'realtime_data.csv'
        ts.get_today_all().to_csv(data_const.My_Store_Dir+realtime)
        print '\n',realtime,'ok'
        wholeindex = 'whole_index.csv'
        ts.get_index().to_csv(data_const.My_Store_Dir+wholeindex)
        print wholeindex,'ok'
        for item,_ in data_const.INDEX_LIST.items():
            prep_d.compute_item(ts.get_k_data(item),'MACD_d').to_csv(data_const.My_Store_Dir+item+data_const.Suffix)
            print item,'ok'
        data_sh_index = pd.read_csv(data_const.My_Store_Dir+'sh.csv')
        last_date = data_sh_index.loc[len(data_sh_index)-1]['date']
        print 'leaving get_last_exchange_day',last_date
        return last_date
    except Exception,e:
        print e
        time.sleep(25)
        return get_last_exchange_day()

#前提:今天是交易日
#功能:更新数据
def update_all():
    #last_exchange_day = getDatetimeToday().strftime("%Y-%m-%d")
    last_exchange_day = get_last_exchange_day()
    basic_env = pd.read_csv(data_const.My_Store_Dir+'basic.csv',index_col='code')
    realtime_data = pd.read_csv(data_const.My_Store_Dir+'realtime_data.csv')
    realtime_data = realtime_data.drop_duplicates((['code']))#去除重复操作十分重要
    realtime_data = realtime_data.set_index('code')
    #用601988.csv中国银行的连续性评估整体文件csv的连续性
    #引入new_code_exist强制更新全部数据
    new_code_exist_force = True#force update
    is_sh_continuous ,_,_ = is_continuous_data('601988', basic_env.loc[601988],last_exchange_day)
    if(not new_code_exist_force and last_exchange_day !=  getDatetimeToday().strftime("%Y-%m-%d") and last_exchange_day!=getDatetimeYesterday().strftime("%Y-%m-%d") and is_sh_continuous):
        print '今日和昨日都不是交易日并且不需要强制更新全部数据，数据无需更新'
        return realtime_data
    if not realtime_data.index.is_unique:
        print '索引重复exception'
        raise
    i=1;codes = data_const.Whole_codes;
    for code in codes:
        print i,'/',len(codes),'update_data',code;i+=1;
        try:
            code_basic = basic_env.loc[int(code)]
            code_rt_dt = realtime_data.loc[int(code)]
        except:
            print 'code_basic = basic_env.loc[int(code)] error'
            continue
        if code_rt_dt['high'].item()==code_rt_dt['low'].item() and code_rt_dt['open'].item()==0 :#停牌处理
            try:
                data = pd.read_csv(data_const.My_Database_Dir+code+data_const.Suffix)
                data = prep_d.preprocess_data(data, code_basic)
                wanted = pd.DataFrame(data[data_const.Feature])
                wanted.to_csv(data_const.My_Database_Dir+code+data_const.Suffix,index=False)
            except Exception as e:
                print e,code
            continue
        isTrue ,last_index,data = is_continuous_data(code, code_basic,last_exchange_day);
        Force_all_renew = True
        if(isTrue and not Force_all_renew):
            try:
                #print 'new_row'
                new_row = idx.new_data_line(last_exchange_day,data.loc[last_index-1],code_rt_dt)
                data.loc[last_index-1,'review_pc'] = code_rt_dt['changepercent']
            except:
                print 'except in new_row'
                new_row = idx.new_data_line(last_exchange_day,data.loc[last_index],code_rt_dt)                
                data.loc[last_index,'review_pc'] = code_rt_dt['changepercent']
            data = data[data_const.Feature]
            data.loc[last_index] = new_row
            #print data.tail()[['review_pc','date','p_change']]
        else: 
            #data = ts.get_hist_data(code).sort_index().reset_index()#有时候有些股票会丢失数据
            print 'is_continuous_data error decided to data=ts.get_k_data(code):',code
            data=ts.get_k_data(code)
            data = prep_d.preprocess_data(data, code_basic)
        wanted = pd.DataFrame(data[data_const.Feature])
        wanted.to_csv(data_const.My_Database_Dir+code+data_const.Suffix,index=False)
    return realtime_data
def is_continuous_data(code,code_basic,last_exchange_day):
    print 'entering is_continuous_data',code,code_basic['name']
    isTrue = False
    try:
        data = pd.read_csv(data_const.My_Database_Dir+code+data_const.Suffix)
        date_index_list = [str(u_date) for u_date in data['date']]
        last_index  =len(date_index_list)-1
        last_date = date_index_list[last_index]
        if len(data)!=len(data.dropna()):raise RuntimeError('data hava nan or something should drop')
        if(getYesterdayDate() in date_index_list):
            isTrue = True
            data = prep_d.preprocess_data(data, code_basic)
        else:
            last_date_tomorrow = getTomorrowDate(last_date)
            if(len(ts.get_k_data(code,last_date_tomorrow,getYesterdayDate()))==0):
                isTrue = True#两个时间之间没有数据，说明数据是连续的
                data = prep_d.preprocess_data(data, code_basic)
    except Exception as e:
        print e,code,'something wrong in is_continuous_data'
        print 'leaving is_continuous_data',isTrue,0,None
        return isTrue,0,None
    if(data.loc[last_index]['date'] !=  last_exchange_day):
        last_index+=1
    print 'leaving is_continuous_data',isTrue,last_index
    return isTrue,last_index,data

if __name__ == '__main__':
    #is_continuous_data('000001',pd.read_csv(data_const.My_Store_Dir+'basic.csv'))
    print '模块测试'
    update_all()
    #print getTomorrowDate('2018-08-11', 1)

