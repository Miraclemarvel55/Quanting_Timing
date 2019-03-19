#!/usr/bin/env pytlhon
# -*- coding: UTF-8 -*-
'''
Created on 2018年8月4日

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

def simu_show(data,ml_func,advice_func,review_num=45,review_end=1):
    data['test'] =0;data.reset_index(drop=True);len_=len(data)
    for i in range(review_end,review_num)[::-1]:
        index = len_-i
        simulation_data = data.head(index)
        #训练函数必须已经内置去除第一条数据的处理，否则这里要自己手动处理
        ml_func_result = ml_func(simulation_data)
        #print 'ml_func_result',ml_func_result
        advice = advice_func(ml_func_result)
        close = simulation_data.tail(1)['close'].item()
        if advice==True:
            advice=close
        else:advice=close/2
        idx = index
        data.loc[idx,'test']=advice
        print simulation_data.tail(1)['date'].item(),close ,advice
    print data
    data = data.set_index('date')
    data = data.tail(review_num-1)[['close','ema5','MACD','test']]
    print data
    data.plot();plt.grid(True);plt.show()

def whole_simulation(data,ml_func,advice_func,review_num=45,review_end=0):
    ishold=0;asset=0;close=0;initial=1;isfirst=0;hold_days=0.1
    #print range(review_end,review_num)[::-1]
    for i in range(review_end,review_num)[::-1]:
        simulation_data = data.head(len(data)-i)
        #训练函数必须已经内置去除第一条数据的处理，否则这里要自己手动处理
        ml_func_result = ml_func(simulation_data)
        #print 'ml_func_result',ml_func_result
        advice = advice_func(ml_func_result)
        close = simulation_data.tail(1)['close'].item()
        if advice:
            isfirst+=1
            if isfirst==1:
                initial=close
        #print  '\n',simulation_data.tail(1)[['date','turnover_rate','MACD_d_d','MACD_d','DIFF_d','ema_quick_d']]
        ishold,asset = single_day_simulation(close, advice, ishold, asset)
        if ishold==1:
            hold_days+=1
        #print i,'advice',advice,'initial',initial,'close',close,' ishold ',ishold,' asset:',asset
    if ishold ==1:
        asset+=close
    hold_rate = (review_num-review_end+0.0)/hold_days
    perform = asset/initial*100*hold_rate
    print '--','initial',initial,'close',close,' ishold ',ishold,' asset:',asset,' perform',perform
    return perform
def single_day_simulation(close=0,advice=True,ishold=0,asset=0):
    if advice:
        if ishold==0:
            ishold=1;asset-=close;
    else :
        if ishold == 1:
            ishold=0;asset+=close
    return ishold,asset
def advice_func_nvbys(temp):
    #print 'entering advice_func_nvbys',temp[0],
    return  not '-' in temp[0]
def advice_func_macd(temp):
    #print 'entering advice_func_macd',temp[0],
    temp = temp[0]
    if temp==ml_const.head or temp == ml_const.body:
        return True
    else: return False
def advice_func_shorterm_fitting(temp):
    #print 'entering advice_func_shorterm_fitting',temp[0]
    return temp[0]    
    
def advice_func_self_exp(temp):
    #print 'entering advice_func_self_exp',temp[0],
    return temp[0]
def advice_func_combine2(temp):
    short =0
    if short==1:
        return advice_func_shorterm_fitting([temp[0]]) and advice_func_nvbys([temp[1]])
    return advice_func_macd([temp[0]]) and advice_func_nvbys([temp[1]])
def advice_func_combine3(temp):
    return advice_func_macd([temp[0]]) and advice_func_nvbys([temp[1]]) and advice_func_self_exp([temp[2]])
def advice_func_combine4(temp):
    return advice_func_macd([temp[0]]) and advice_func_nvbys([temp[1]]) and advice_func_self_exp([temp[2]]) and advice_func_shorterm_fitting([temp[3]])
def combine2_simulation(codata,ml1=ml_all.macd_diff_judge,ml2=ml_all.naive_bayes):
    short=0
    if short==1:
        ml1=ml_all.shorterm_fitting
    return ml1(codata)+ml2(codata)
def combine3_simulation(codata,ml1=ml_all.macd_diff_judge,ml2=ml_all.naive_bayes,ml3=ml_all.self_experiences_judge):
    return ml1(codata)+ml2(codata)+ml3(codata)
def combine4_simulation(codata,ml1=ml_all.macd_diff_judge,ml2=ml_all.naive_bayes,ml3=ml_all.self_experiences_judge,ml4=ml_all.shorterm_fitting):
    return ml1(codata)+ml2(codata)+ml3(codata)+ml4(codata)

#近n日大盘指标涨跌幅度
def getlast_n_days_p_change(code='sz50',n=55,end=0):
    codata = pd.read_csv(dtcst.My_Store_Dir+code+dtcst.Suffix)
    last_n_close = codata.tail(n).head(n-end)['close']
    last_n_days_p_change = (last_n_close.tail(1).item()-last_n_close.head(1).item())/last_n_close.head(1).item()*100
    return last_n_days_p_change
def whole_stock_ml_perform_simulation(review_num=45,review_end=0,codes=dtcst.Whole_codes,not_want_codes_list=[],test_method=3):
    if len(codes)==0:
        print 'entering whole_stock_ml_perform_simulation: ',review_num,review_end,'not_wanted_codes_list:',len(not_want_codes_list),\
        ':',not_want_codes_list,'\nwanted_codes:',len(codes)
        return [],[],0
    perform  = [];not_zero_profit_dic={};
    i=1;#codes = list(set(codes).difference(set((util.myflatten(not_want_codes_list)))))
    print 'entering whole_stock_ml_perform_simulation: ',review_num,review_end,'not_wanted_codes_list:',len(not_want_codes_list),\
        ':',not_want_codes_list,'\nwanted_codes:',len(codes)
    len_codes = len(codes)
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
        if test_method=='naive_bayes':
            code_perform = whole_simulation(codata,ml_all.naive_bayes,advice_func_nvbys,review_num,review_end)
        elif test_method=='macd_diff_judge':
            code_perform = whole_simulation(codata,ml_all.macd_diff_judge,advice_func_macd,review_num,review_end)
        elif test_method=='shorterm_fitting':
            code_perform = whole_simulation(codata,ml_all.shorterm_fitting,advice_func_shorterm_fitting,review_num,review_end)
        elif test_method==2:
            code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀
        elif test_method==3:
            code_perform = whole_simulation(codata,combine3_simulation,advice_func_combine3,review_num,review_end)
        elif test_method==4:
            code_perform = whole_simulation(codata,combine4_simulation,advice_func_combine4,review_num,review_end)
        else:
            print 'default test_method code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀'
            code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀
        perform.append(code_perform)
        if code_perform!=0:
            not_zero_profit_dic[code]=code_perform
        #perform.append(whole_simulation(codata,combine4_simulation,advice_func_combine4,55))
        print '                         code',code,i-1,'/',len_codes,'simulation'
    print '\nperform',perform
    up =[i for i in perform if i>0];down = [i for i in perform if i<0]
    sum_up = sum(up);sum_down = sum(down);len_pf = len(perform); no_zero_profit_num=(len(up)+len(down))
    print '收益大于0的投资股票个数',len(up),'/',len_pf,'收益:',sum_up
    print '收益等于0的投资股票个数',len([i for i in perform if i==0]),'/',len_pf
    print '收益小于0的投资股票个数',len(down),'/',len_pf,'损失:',sum_down
    try:
        profit_loss = (sum_up+sum_down)/no_zero_profit_num
    except:profit_loss=0
    print '总的盈利情况: （收益+损失)/非零收益总个数: ',(sum_up+sum_down),'/',no_zero_profit_num,'=',profit_loss  
    print '同期指数收益--',
    for index_dic_key,index_dic_value in dtcst.INDEX_LIST.items():
        print index_dic_value,':',getlast_n_days_p_change(index_dic_key,review_num,review_end),
    #values_list = not_zero_profit_dic.values()
    up_values_list=[];down_values_list=[];down_value_codes=[]
    for key,value in not_zero_profit_dic.items():
        if value>0:
            up_values_list.append(value)
        elif value<0:
            down_value_codes.append(key)
            down_values_list.append(value)
    print ''
    if len(not_zero_profit_dic)==0:
        print 'exception not_zero_profit_dic codes dic size=0'
        return [],[],0
    try:
        dislike_down_mean = sum(down_values_list)/len(down_values_list)
        dislike_up_mean = sum(up_values_list)/len(up_values_list)
    except:
        dislike_down_mean = 0
        dislike_up_mean = 0
    dislike_codes=[];dislike_codes2=[];dislike_codes3=[];dislike_down_thresh=dislike_down_mean/2;dislike_up_thresh = dislike_up_mean*1.5
    for key,value in not_zero_profit_dic.items():
        if value<=dislike_down_mean:
            dislike_codes.append(key)
        elif 0>value>dislike_down_thresh:
            pass
            #dislike_codes2.append(key)
        elif dislike_down_thresh>=value>dislike_down_mean:
            dislike_codes3.append(key)
    #print 'dislike_codes:',len(dislike_codes),dislike_codes
    mean_target=0;list_dimension_not1=1
    for not_want_list in not_want_codes_list:
        if util.isNum(not_want_list):
            mean_target = intersection(not_want_codes_list, down_value_codes)
            list_dimension_not1 = 0
            break
        mean_target +=intersection(not_want_list,down_value_codes)
    if list_dimension_not1==1 and len(not_want_codes_list)!=0:
        mean_target = mean_target/len(not_want_codes_list)
    return [dislike_codes3,dislike_codes2,dislike_codes],down_value_codes,mean_target
def intersection(not_want_list,down_value_codes):
    mean_target=0
    if len(not_want_list)*len(down_value_codes)!=0:
            intersection_list_codes = list(set(not_want_list).intersection(set(down_value_codes)))
            intersectoin_proportion = (len(intersection_list_codes)+0.0)/len(down_value_codes)
            print 'codes intersectoin_proportion:',len(intersection_list_codes),'/',len(down_value_codes), '=',intersectoin_proportion,\
                len(intersection_list_codes),'/',len(not_want_list), '=',(len(intersection_list_codes)+0.0)/len(not_want_list)
            mean_target = intersectoin_proportion/len(not_want_list)*len(intersection_list_codes)
            print 'mean target codes:',mean_target
    return mean_target
def get_optimized_period_review_num_of_simulation(codes=dtcst.Whole_codes):
    period_up_thresh = 56;period_down_thresh = 5;wanted_period=5;
    num_list = range(period_down_thresh,period_up_thresh);print num_list
    from collections import OrderedDict
    period_perform = OrderedDict()
    _,down_value_codes,_ = whole_stock_ml_perform_simulation(review_num=wanted_period,codes=codes)
    for period in num_list:
        print 'period',period
        not_want_codes_list,_,_ =whole_stock_ml_perform_simulation(review_num=wanted_period+period,review_end=wanted_period,codes=codes)
        not_want_list = util.myflatten(not_want_codes_list)
        mean_target = intersection(not_want_list, down_value_codes)
        period_perform[period]=mean_target
        print 'period',period,'perform:',mean_target
    print period_perform
    for key,value in period_perform.items():
        print key,':',value,',',
    key_max_wanted=0;max_wanted=-10;
    for key,value in period_perform.items():
        wanted=value
        if wanted>max_wanted:
            max_wanted=wanted;key_max_wanted=key
    print 'key_max_wanted',key_max_wanted,max_wanted
    return key_max_wanted

def get_optimized_func_of_codes(review_num=45,review_end=0,codes=dtcst.Whole_codes):
    if len(codes)==0:
        print 'entering get_optimized_func_of_codes: ',review_num,review_end
        return [],[],0
    i=1;print 'entering get_optimized_func_of_codes: ',review_num,review_end
    len_codes = len(codes)
    test_methods = ['naive_bayes','macd_diff_judge','shorterm_fitting',2]
    #test_methods = ['macd_diff_judge','shorterm_fitting']
    code_func_map = {}
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
        max_perform =-999;max_func=2
        for test_method in test_methods:
            print test_method,
            if test_method=='naive_bayes':
                code_perform = whole_simulation(codata,ml_all.naive_bayes,advice_func_nvbys,review_num,review_end)
            elif test_method=='macd_diff_judge':
                code_perform = whole_simulation(codata,ml_all.macd_diff_judge,advice_func_macd,review_num,review_end)
            elif test_method=='shorterm_fitting':
                code_perform = whole_simulation(codata,ml_all.shorterm_fitting,advice_func_shorterm_fitting,review_num,review_end)
            elif test_method==2:
                code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀
            elif test_method==3:
                code_perform = whole_simulation(codata,combine3_simulation,advice_func_combine3,review_num,review_end)
            elif test_method==4:
                code_perform = whole_simulation(codata,combine4_simulation,advice_func_combine4,review_num,review_end)
            else:
                print 'default test_method code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀'
                code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀
            if max_perform<=code_perform:
                max_perform=code_perform
                max_func = test_method
        code_func_map[code]=max_func
    print code_func_map
    import marshal 
    output_file = open(dtcst.Project_Root_Dir+"code_func_map",'wb')#把这些数据序列化到文件中，注：文件必须以二进制模式打开
    marshal.dump(code_func_map,output_file)
    output_file.close()
    print '\ncode_func_map write over'
    
def whole_stock_ml_perform_simulation_of_func_code(review_num=45,review_end=0,codes=dtcst.Whole_codes,not_want_codes_list=[]):
    if len(codes)==0:
        print 'entering whole_stock_ml_perform_simulation_of_func_code: ',review_num,review_end
        return [],[],0
    perform  = [];not_zero_profit_dic={};
    i=1;print 'entering whole_stock_ml_perform_simulation_of_func_code: ',review_num,review_end
    len_codes = len(codes)
    try:
        import marshal
        input_file = open(util.findPath('Project_Root_File.ini')+'code_func_map','rb')#从文件中读取序列化的数据
        #data1 = []
        code_func_map = marshal.load(input_file)
    except Exception,e:
        print e
        print 'marshal error'
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
        try:
            test_method = code_func_map[code]
        except: 
            test_method = 2
        print test_method,':',
        if test_method=='naive_bayes':
            code_perform = whole_simulation(codata,ml_all.naive_bayes,advice_func_nvbys,review_num,review_end)
        elif test_method=='macd_diff_judge':
            code_perform = whole_simulation(codata,ml_all.macd_diff_judge,advice_func_macd,review_num,review_end)
        elif test_method=='shorterm_fitting':
            code_perform = whole_simulation(codata,ml_all.shorterm_fitting,advice_func_shorterm_fitting,review_num,review_end)
        elif test_method==2:
            code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀
        elif test_method==3:
            code_perform = whole_simulation(codata,combine3_simulation,advice_func_combine3,review_num,review_end)
        elif test_method==4:
            code_perform = whole_simulation(codata,combine4_simulation,advice_func_combine4,review_num,review_end)
        else:
            print 'default test_method code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀'
            code_perform = whole_simulation(codata,combine2_simulation,advice_func_combine2,review_num,review_end)#优秀
        perform.append(code_perform)
        if code_perform!=0:
            not_zero_profit_dic[code]=code_perform
        #perform.append(whole_simulation(codata,combine4_simulation,advice_func_combine4,55))
        print '                         code',code,i-1,'/',len_codes,'simulation'
    print '\nperform',perform
    up =[i for i in perform if i>0];down = [i for i in perform if i<0]
    sum_up = sum(up);sum_down = sum(down);len_pf = len(perform); no_zero_profit_num=(len(up)+len(down))
    print '收益大于0的投资股票个数',len(up),'/',len_pf,'收益:',sum_up
    print '收益等于0的投资股票个数',len([i for i in perform if i==0]),'/',len_pf
    print '收益小于0的投资股票个数',len(down),'/',len_pf,'损失:',sum_down
    try:
        profit_loss = (sum_up+sum_down)/no_zero_profit_num
    except:profit_loss=0
    print '总的盈利情况: （收益+损失)/非零收益总个数: ',(sum_up+sum_down),'/',no_zero_profit_num,'=',profit_loss  
    print '同期指数收益--',
    for index_dic_key,index_dic_value in dtcst.INDEX_LIST.items():
        print index_dic_value,':',getlast_n_days_p_change(index_dic_key,review_num,review_end),
    #values_list = not_zero_profit_dic.values()
    up_values_list=[];down_values_list=[];down_value_codes=[]
    for key,value in not_zero_profit_dic.items():
        if value>0:
            up_values_list.append(value)
        elif value<0:
            down_value_codes.append(key)
            down_values_list.append(value)
    print ''
    if len(not_zero_profit_dic)==0:
        print 'exception not_zero_profit_dic codes dic size=0'
        return [],[],0
    try:
        dislike_down_mean = sum(down_values_list)/len(down_values_list)
        dislike_up_mean = sum(up_values_list)/len(up_values_list)
    except:
        dislike_down_mean = 0
        dislike_up_mean = 0
    dislike_codes=[];dislike_codes2=[];dislike_codes3=[];dislike_down_thresh=dislike_down_mean/2;dislike_up_thresh = dislike_up_mean*1.5
    for key,value in not_zero_profit_dic.items():
        if value<=dislike_down_mean:
            dislike_codes.append(key)
        elif 0>value>dislike_down_thresh:
            pass
            #dislike_codes2.append(key)
        elif dislike_down_thresh>=value>dislike_down_mean:
            dislike_codes3.append(key)
    #print 'dislike_codes:',len(dislike_codes),dislike_codes
    mean_target=0;list_dimension_not1=1
    for not_want_list in not_want_codes_list:
        if util.isNum(not_want_list):
            mean_target = intersection(not_want_codes_list, down_value_codes)
            list_dimension_not1 = 0
            break
        mean_target +=intersection(not_want_list,down_value_codes)
    if list_dimension_not1==1 and len(not_want_codes_list)!=0:
        mean_target = mean_target/len(not_want_codes_list)
    return [dislike_codes3,dislike_codes2,dislike_codes],down_value_codes,mean_target
def simu_show_main(ml_func=ml_all.naive_bayes,advice_func=advice_func_nvbys,review_num=45,review_end=1):
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

def simulation():
    review_num=30;review_end = review_num/2
    codes=dtcst.Whole_codes
    if dtcst.RunningOnServer:
            import random
            codes = random.sample(codes, len(codes)/100)
    #simu_show_main(ml_func=ml_all.shorterm_fitting,advice_func=advice_func_shorterm_fitting,review_num=review_num)
    #simu_show_main(ml_func=ml_all.macd_diff_judge,advice_func=advice_func_macd,review_num=review_num)
    #simu_show_main(combine2_simulation, advice_func_combine2, review_num)
    get_optimized_func_of_codes(review_num, review_end,codes=codes)
    whole_stock_ml_perform_simulation_of_func_code(review_num=review_end,codes=codes)

if __name__ == '__main__':
    review_num=30;review_end = review_num/2
    codes=dtcst.Whole_codes
    if dtcst.RunningOnServer:
            import random
            codes = random.sample(codes, len(codes)/100)
    #simu_show_main(ml_func=ml_all.shorterm_fitting,advice_func=advice_func_shorterm_fitting,review_num=review_num)
    #simu_show_main(ml_func=ml_all.macd_diff_judge,advice_func=advice_func_macd,review_num=review_num)
    #simu_show_main(combine2_simulation, advice_func_combine2, review_num)
    get_optimized_func_of_codes(review_num, review_end,codes=codes)
    whole_stock_ml_perform_simulation_of_func_code(review_num=review_end,codes=codes)