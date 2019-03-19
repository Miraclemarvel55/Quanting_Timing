#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import util,sys
"""功能：        
放置常量，列表    
"""
Billion_1_10 = 1000*1000*100
HS_stock_code_len=6
H_code_base=600000 #沪市股票代码基位
H_code_ceiling=604444 #沪市股票代码基本上达不到位
S_code_Base=000000  #深市股票代码基位
S_code_ceiling=3333  #深市股票代码基本上达不到位
#沪深两市股票所有可能的代码----passed
HS_stock_codes=[str(item) for item in list(range( H_code_base, H_code_ceiling))]+util.intstr_list_zfill([str(i)for i in list(range( S_code_Base, S_code_ceiling))],HS_stock_code_len)
Suffix = '.csv'
Project_Root_Dir = util.findPath('Project_Root_File.ini')
Store_Dir=Project_Root_Dir+'resources/stock_data/'
Test_Store_Dir = Project_Root_Dir+'resources/Test_stockdata/'
Database_Dir=Store_Dir + 'get_k_data/'
Test_Database_Dir = Test_Store_Dir + 'testdata/'
INDEX_LIST = {'sh':'沪指涨跌幅度','sz':'深证成指','hs300':'沪深300','sz50':'上证50'}
Meta_Index_Needing = ['date','open','high','close','low','volume']#nmc 流通市值
Process_Index_Needing_whole = ['nmc','amount','turnover_rate','p_change','review_pc','amplitude','ema5','MACD','DIFF','DEM',\
                         'ema_quick','ema_slow','K','D','J','RSV','high_d','low_d','ema5_d','MACD_d','DIFF_d','DEM_d','K_d','D_d',\
                         'ema_quick_d','ema_slow_d','MACD_d_d','ema5_d_d','efficiency_coefficient','close_ama','close_ama_d','backward_inference_MACD_hit',\
                         'is_break_up_n_high_or_low','number_stationary_in_n_days','is_d_bigger_than_d_mean','continuous_up_down_day',\
                         'big_drop_rise_review_days','volume_pc_feature','recommend_period','fear_capital_feature','greedy_capital_feature',\
                         'me_tooism_human_feature','fear_human_feature','feeble_feature','positive_volume_sum','positive_v_continuous_level',\
                         'negative_volume_sum','negative_v_continuous_level','positive_v_continuous_portion_nega_sum','negative_v_continuous_portion_posi_sum',\
                         'PVT','ema5_PVT','ema5_PVT_d']

Process_Index_Needing = Process_Index_Needing_whole
Derivative_str = '_d'  
Feature = Meta_Index_Needing + Process_Index_Needing
Time_Related_Index = ['volume','amount','turnoverratio']
K_TYPE_KEY = ['D', 'W', 'M']
K_TYPE_MIN_KEY = ['5', '15', '30', '60']
K_TYPE = {'D': 'k_daily', 'W': 'k_weekly', 'M': 'k_monthly'}
Test_Wanted_codes = ['000001','603603','603032','600340','600036','002219','603259','600036','603590','600759','601988','601577','603969','002263','002489','002161','000895','002357','002568','002227','002254','002189','603725','000502','002417','002312','603598','603339','002546','002091']
My_Database_Dir = Test_Database_Dir
My_Store_Dir = Test_Store_Dir
Whole_codes = Test_Wanted_codes
My_Wanted_codes = list(set(Test_Wanted_codes).difference(set(util.not_want_codes_list())));#Test_Wanted_codes'600666',
#辅助判断在服务器32位运行True，还是本地64位运行False
RunningOnServer = not bool(sys.maxsize> 2**32)
#RunningOnServer = True
if RunningOnServer:
    My_Database_Dir = Database_Dir
    My_Store_Dir = Store_Dir
    Whole_codes=util.get_wanted_codes()
    #Whole_codes = ['603603']
    if True:
        '''import random
        Whole_codes = random.sample(Whole_codes, len(Whole_codes)/10)'''
        Whole_codes = Whole_codes[:283]
    My_Wanted_codes = list(set(Whole_codes).difference(set(util.not_want_codes_list())));
