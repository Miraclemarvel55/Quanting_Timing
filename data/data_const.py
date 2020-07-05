#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import util,sys,os
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
# Project_Root_Dir = "/home/feiyu/ProjectSave/SAP_Python_0/"
all_project_root_path=os.path.dirname(os.path.dirname(Project_Root_Dir))+'/';
cweb_path=all_project_root_path+"StockServer/CWebServer/webserver/"
cweb_analyzer_path = os.path.join(cweb_path,"")
Store_Dir=Project_Root_Dir+'resources/stock_data/'
Test_Store_Dir = Project_Root_Dir+'resources/Test_stockdata/'
Database_Dir=Store_Dir + 'get_k_data/'
Test_Database_Dir = Test_Store_Dir + 'testdata/'
INDEX_LIST = {'sh':'沪指涨跌幅度','sz':'深证成指','hs300':'沪深300','sz50':'上证50'}
Meta_Index_Needing = ['date','open','high','close','low','volume']#nmc 流通市值
Process_Index_Needing_whole = ['nmc','amount','turnover_rate','p_change','review_pc','amplitude','efficiency_coefficient',\
                         "dist2last_big_drop",'PVT','stationary_info','dist2last_min','dist2last_max',\
                         'ema3','ema3_p_change','ema3_PVT','ema3_stationary_info','ema3_dist2last_min','ema3_dist2last_max',\
                         'ema5','ema5_p_change','ema5_PVT','ema5_stationary_info','ema5_dist2last_min','ema5_dist2last_max',\
                         'pseudo_mean_price','pseudo_mean_price_p_change','pseudo_mean_price_PVT','pseudo_mean_price_stationary_info','pseudo_mean_price_dist2last_min','pseudo_mean_price_dist2last_max',\
                         'close_ama','close_ama_p_change','close_ama_PVT','close_ama_stationary_info','close_ama_dist2last_min','close_ama_dist2last_max',\
                         "MACD","MACD_d","volume_d","backward_inference_MACD_hit",
                         ]

Process_Index_Needing = Process_Index_Needing_whole
Derivative_str = '_d'  
Feature = Meta_Index_Needing + Process_Index_Needing
Time_Related_Index = ['volume','amount','turnoverratio']
K_TYPE_KEY = ['D', 'W', 'M']
K_TYPE_MIN_KEY = ['5', '15', '30', '60']
K_TYPE = {'D': 'k_daily', 'W': 'k_weekly', 'M': 'k_monthly'}
Test_Wanted_codes =["002432"]# ["601678","000568","600883","600513","002932","000601","603035","002776","603992","600802","002583","002415","603879","002456","002052","002413","603992","603879","600802","002456","002079","603005",'002215','603906','601222','002269','600505','600626','000929','601066','002952','601615','601865','002953','002250','002360','603877','000716','000001','603603','603032','600340','600036','002219','603259','600036','603590','600759','601988','601577','603969','002263','002489','002161','000895','002357','002568','002227','002254','002189','603725','000502','002417','002312','603598','603339','002546','002091']
My_Database_Dir = Test_Database_Dir
My_Store_Dir = Test_Store_Dir
Whole_codes = Test_Wanted_codes
My_Wanted_codes = list(set(Test_Wanted_codes).difference(set(util.not_want_codes_list())));#Test_Wanted_codes'600666',
#辅助判断在服务器32位运行True，还是本地64位运行False
RunningOnServer = not "feiyu" in os.environ["HOME"] or "sida" in os.environ["HOME"]
if RunningOnServer:
    My_Database_Dir = Database_Dir
    My_Store_Dir = Store_Dir
    Whole_codes=util.get_wanted_codes()
    #Whole_codes = Whole_codes[:2]+['000620']

    My_Wanted_codes = list(set(Whole_codes).difference(set(util.not_want_codes_list())))
