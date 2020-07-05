#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import data.data_const as dtcst
import data.util as util

import warnings
warnings.filterwarnings('ignore')

def whole_simulation(data,ml_func,advice_func,review_num=45,review_end=0,accerating=True):
    ishold=0;asset=0;close=0;initial=1;isfirst=0;hold_days=0
    if accerating==True:
        ml_func_results = ml_func(data,review_num=review_num,review_end=review_end)
        for index, item in ml_func_results.iterrows():
            #print index,item,type(item)
            close = item['close']
            ml_func_result = item.drop('close').tolist()
            #print 'ml_func_result',ml_func_result
            advice = advice_func(ml_func_result)        
            if advice:
                isfirst+=1
                if isfirst==1:#初始化资产
                    initial=close
            #print  '\n',simulation_data.tail(1)[['date','turnover_rate','MACD_d_d','MACD_d','DIFF_d','ema_quick_d']]
            ishold,asset = single_day_simulation(close, advice, ishold, asset,None)
            #if ishold==1:
                #hold_days+=1
            #print index,'advice',advice,'initial',initial,'close',close,' ishold ',ishold,' asset:',asset
    else:
        for i in range(review_end,review_num)[::-1]:
            simulation_data = data.head(len(data)-i)
            #训练函数必须已经内置去除第一条数据的处理，否则这里要自己手动处理
            ml_func_result = ml_func(simulation_data)
            advice = advice_func(ml_func_result)
            tail_data2=simulation_data.tail(2)#;today_data=tail_data2.tail(1);old_data=tail_data2.head(1)
            close = tail_data2.tail(1)['close'].item()
            advice = advice_func(ml_func_result)        
            if advice:
                isfirst+=1
                if isfirst==1:#初始化资产
                    initial=close
            ishold,asset = single_day_simulation(close, advice, ishold, asset,None)
            if ishold==1:
                hold_days+=1
    if ishold ==1:
        asset+=close
    hold_rate = hold_days/(review_num-review_end+0.0)
    perform = asset/initial*100;stubborn =(close-data.iloc[-review_num]['close'].item())/initial*100;alpha=perform-stubborn
    print 'initial',initial,'close',close,' ishold ',ishold,' asset:',asset,'hold_rate',hold_rate,'perform',perform,'stubborn',stubborn,\
    '\n','alpha',alpha,'start',data.iloc[-review_num]['date'],"end",data.iloc[-(review_end+1)]['date']
    return perform
def single_day_simulation(close=0,advice=True,ishold=0,asset=0,env=None):
    switch_price_predict = False;
    if switch_price_predict:
        last_date_data=env.head(1);#此时env只有两条数据，昨日和今日
        old_high=last_date_data['high'].item();
        old_p_change=last_date_data['p_change'].item();old_close=last_date_data['close'].item();
        predict_price=0;today_high=env.tail(1)['high'].item()
        if old_p_change<=1.5:predict_price=old_high*0.995;
        else:predict_price=old_close*(1+old_p_change*1.5/100)
        
        if advice:
            if ishold==0:
                ishold=1;asset-=close;
            else:
                if today_high>=predict_price:
                    asset+=predict_price-close
        else :
            if ishold == 1:
                if today_high>=predict_price:
                    asset+=predict_price
                else:asset+=close
                ishold=0;
    else:        
        if advice:
            if ishold==0:
                ishold=1;asset-=close;
        else :
            if ishold == 1:
                asset+=close;ishold=0;
    return ishold,asset

def advice_func_general(temp,and_or_logic_list=[]):
    aoll = and_or_logic_list
    if temp :#temps为True或者不空列表
        if temp==True:
            return True
        elif aoll:
            identity_element = True#幺元
            '''for tp in temp:
                identity_element = identity_element and tp'''
            for (t,aol) in zip(temp,aoll):
                #print t,aol
                identity_element=aol(identity_element,t)
            #print temp,aoll,identity_element           
            return identity_element
    else : 
        return False
getlast_n_days_p_change = util.getlast_n_days_p_change
#单一方法或局部整合方法测试及其实现函数
def whole_stock_ml_perform_simulation(review_num=45,review_end=0,codes=dtcst.Whole_codes,not_want_codes_list=[],test_method='ml_svm_predict'):
    if len(codes)==0:
        print 'entering whole_stock_ml_perform_simulation: ',review_num,review_end,'not_wanted_codes_list:',len(not_want_codes_list),\
        ':',not_want_codes_list,'\nwanted_codes:',len(codes)
        return [],[],0
    perform  = [];not_zero_profit_dic={};
    i=1;#codes = list(set(codes).difference(set((util.myflatten(not_want_codes_list)))))
    print 'entering whole_stock_ml_perform_simulation: ',review_num,review_end,'not_wanted_codes_list:',len(not_want_codes_list),\
        ':',not_want_codes_list,'\nwanted_codes:',len(codes)
    len_codes = len(codes)
    #使得大盘日期与个股日期回测日期映射保持日期一致性
    review_num_date,review_end_date  = util.get_coherence_date(review_num, review_end)
    review_num_cp = review_num;review_end_cp = review_end
    for code in codes:
        review_num = review_num_cp;review_end = review_end_cp;
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
        #对review_num 和 review_end 重新映射使得满足和大盘日期一致性
        if False:
            review_num,review_num_date_or_wanted= util.review_num_end_mapping(codata,review_num_date,review_end_date,which_want_or_change_direction=1)
            review_end,review_end_date= util.review_num_end_mapping(codata,review_end_date,review_num_date_or_wanted,which_want_or_change_direction=-1)
        if review_num==None or review_end==None:continue
        review_num+=1
        from ml_stock import ml_all
        code_perform = whole_simulation(codata,ml_all.func_name_address[test_method],advice_func_general,review_num,review_end,accerating=False)
        perform.append(code_perform)
        if code_perform!=0:
            not_zero_profit_dic[code]=code_perform
        #perform.append(whole_simulation(codata,combine4_simulation,advice_func_combine4,55))
        print '                         code',code,i-1,'/',len_codes,'simulation'
    print '\n',test_method,'perform',perform
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
        print index_dic_value,':',getlast_n_days_p_change(codata = None,code=index_dic_key,n=review_num_cp,end=review_end_cp),
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
#单一ml_func codes数据集测试
def single_ml_method_simu(test_methods=['default']):
    codes=dtcst.Whole_codes
    review_nums = [55]
    for review_num in review_nums:
        for test_method in test_methods:
            print 'perform simulation 收益 of review_num',review_num,test_method
            whole_stock_ml_perform_simulation(review_num, codes=codes,test_method=test_method)

#将数据和ml_func的输出标签结合，用于func组合时避免指数次调用func，只需要组合相应的标签输出
def advice_fill(label,data,ml_func,review_num,review_end):
    data[label] =False;
    data.reset_index(drop=True);len_=len(data)
    for i in range(review_end,review_num)[::-1]:
        index = len_-i#注意head()和data.loc() 参数含义的不同head参数表示前第几行，loc参数表示索引
        simulation_data = data.head(index)
        #训练函数必须已经内置去除第一条数据的处理，否则这里要自己手动处理
        #print i,simulation_data,index,len_
        advice = ml_func(simulation_data)
        idx = index-1
        data.loc[idx,label]=advice
    return data
#得到最优的codes 股票的拟合ml_func
def get_optimized_func_of_codes(review_num=45,review_end=0,codes=dtcst.Whole_codes):
    if len(codes)==0:
        print ' get_optimized_func_of_codes error: codes.lenth=0',review_num,review_end
        return None,None
    i=1;print 'entering get_optimized_func_of_codes: ',review_num,review_end
    len_codes = len(codes)
    from ml_stock import ml_all
    func_name_address = ml_all.func_name_address
    func_list = func_name_address;func_list_combinations=[];
    #搭配ml算法和逻辑测验单个1分钟极限在C4/7 × 3**3=945 1k量级 :总共7个算法最高4个组合一起，留3个坑位插入逻辑运算
    from itertools import combinations;limit_coopers=5
    if len(func_list)<=limit_coopers:
        max_coopers=len(func_list)
    else:max_coopers = limit_coopers
    cooperations_num = range(1,max_coopers);
    for itemp in cooperations_num:
        func_list_combinations+=list(combinations(func_list, itemp))
    test_methods = func_list_combinations
    aol_func_map = util.aol_func_map
    code_func_map = {}
    #使得大盘日期与个股日期回测日期映射保持日期一致性
    review_num_date,review_end_date  = util.get_coherence_date(review_num, review_end)
    review_num_cp = review_num;review_end_cp = review_end
    for code in codes:
        review_num = review_num_cp;review_end = review_end_cp;
        print i,'/',len_codes,'simulation',code;
        try:
            codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
        except IOError,e:
            print e;i-=1;len_codes-=1
            continue
        if len(codata)<=review_num+25:
            print 'data amount too small'
            i-=1;len_codes-=1
            code_func_map[code] = [func_name_address.keys()[:3],aol_func_map.keys()[:3]]#随便给的,数据量要求小的ml算法
            print code_func_map[code]
            continue
        #对review_num 和 review_end 重新映射使得满足和大盘日期一致性
        review_num,review_num_date_or_wanted= util.review_num_end_mapping(codata,review_num_date,review_end_date,which_want_or_change_direction=1) 
        review_end,review_end_date= util.review_num_end_mapping(codata,review_end_date,review_num_date_or_wanted,which_want_or_change_direction=-1)
        if review_num==None or review_end==None:
            code_func_map[code] = [func_name_address.keys()[:3],aol_func_map.keys()[:3]]#随便给的,数据量要求小的ml算法
            continue
        review_num+=1
        data = codata
        for name,ml_func in func_name_address.iteritems():#预填充数据，避免下面指数级别重复计算
            data = advice_fill(name, data, ml_func, review_num, review_end)
        codata = data
        max_perform =-999;max_func= ['default']
        max_end=len(test_methods[-1])
        fun_map_keys=aol_func_map.keys()
        def reproducible_permutation_all(elements,slots):#可重复排列函数elements被组合元素，slots组合的坑数---孔数
            slots+1;#确保slots是数字
            if slots==0:return [[]]
            else: 
                Y=[];
                for X in reproducible_permutation_all(elements,slots-1):
                    X=tuple(X)
                    for element in elements:
                        temp=list(X);
                        temp.append(element);
                        Y.append(temp);
                return Y
        for test_method in test_methods:
            #函数代理，每次可以迭代使用不同的函数或者函数组合，闭包外参数为不同的函数字典test_method           
            def ml_all_agent(codata,test_method=test_method,review_num=None,review_end=None):
                if review_num==review_end==None:#单次迭代函数
                    last_data = codata.tail(1)
                    return [last_data[name].item() for name in test_method]
                else :return codata.head(len(codata)-review_end).tail(review_num-review_end)[list(test_method)+['close']]
            and_or_logic_lists = reproducible_permutation_all(fun_map_keys,len(test_method)-1)
            for aol in and_or_logic_lists:
                aol.insert(0,'identity_func')#因为aol 里面只有列表推导的中间函数，需要补充一个幺元函数初始化
                real_aol = [aol_func_map[key] for key in aol]
                #不同的标签与或组合代理函数，实现不同标签能力的互补，或者制约，并且或逻辑可以增加逻辑结果为True连续性
                def advice_func_special(temp,and_or_logic_list=real_aol):
                    return advice_func_general(temp, and_or_logic_list)
                print '\n',i,'/',len_codes,'simulation_code_func_method choose:',code,len(test_method),'/',max_end ,test_method,aol;
                code_perform = whole_simulation(codata,ml_all_agent,advice_func_special,review_num,review_end)
                if max_perform<=code_perform:
                    max_perform=code_perform
                    max_func = [test_method,aol]
                    print '循环至今最大perform',max_perform,'本次perform',code_perform,'max函数地址',max_func,'本次函数地址',test_method,'\n'
        code_func_map[code]=max_func
        print '最大perform 确认：',max_perform;util.type_data(code_func_map[code]),'\n'
        i+=1;
    print '\n',code_func_map
    return code_func_map,func_name_address

#codes股票的最优ml_func回测，及结果，并且生成下次需要剔除的not_like_codes 
#若输入的最优ml_func回测为NOne，则在线计算出最优回测
def whole_stock_ml_perform_simulation_of_func_code(review_num=45,review_end=0,codes=dtcst.Whole_codes,not_want_codes_list=[],code_func_map=None,func_name_address=None):
    if len(codes)==0:
        print ' whole_stock_ml_perform_simulation_of_func_code error : len(codes)=0',review_num,review_end
        return [],[],0,0
    perform  = [];not_zero_profit_dic={};
    i=1;print 'entering whole_stock_ml_perform_simulation_of_func_code: ',review_num,review_end
    len_codes = len(codes)
    try:
        if code_func_map==None:
            code_func_map = util.code_func_map()
    except Exception,e:
        print e
        print 'marshal error'
        code_func_map,func_name_address = get_optimized_func_of_codes(review_num+20,review_end+20,codes)
    #使得大盘日期与个股日期回测日期映射保持日期一致性
    review_num_date,review_end_date  = util.get_coherence_date(review_num, review_end)
    review_num_cp = review_num;review_end_cp = review_end
    for code in codes:
        review_num = review_num_cp;review_end = review_end_cp;
        print '\n',i,'/',len_codes,'simulation',code;i+=1;
        try:
            codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
        except IOError,e:
            print e;i-=1;len_codes-=1
            continue
        if len(codata)<=review_num*1.5:
            print 'data amount too small'
            i-=1;len_codes-=1
            continue
        #对review_num 和 review_end 重新映射使得满足和大盘日期一致性
        review_num,review_num_date_or_wanted= util.review_num_end_mapping(codata,review_num_date,review_end_date,which_want_or_change_direction=1) 
        review_end,review_end_date= util.review_num_end_mapping(codata,review_end_date,review_num_date_or_wanted,which_want_or_change_direction=-1)
        if review_num==None or review_end==None:continue
        review_num+=1
        try:
            from ml_stock import ml_all
            if func_name_address==None:
                func_name_address = ml_all.func_name_address
            test_method = [func_name_address[name]for name in code_func_map[code][0]]#0代表是ml_method 控制
        except: 
            print 'whatever fix single code_func_map[code] ----whole_stock_ml_perform_simulation_of_func_code'
            import ml_stock.not_want_codes_list_object_store as maker
            code_func_map_single = maker.not_want_codes_list_and_code_func_map_generator(issingle_train=True,code=code,review_num=review_num+20,review_end=review_end+20)
            code_func_map[code]=code_func_map_single
            code_funcs = code_func_map[code][0]
            test_method = [func_name_address[name]for name in code_funcs]#0代表是ml_method 控制
            print 'whole_stock_ml_perform_simulation_of_func_code get code_func_map value error but fixed by single train'
        print test_method
        combination_func = lambda codata: [func(codata) for func in test_method]
        aol_func_map =util.aol_func_map
        try:
            real_aol = [aol_func_map[key] for key in code_func_map[code][1]]
        except: 
            real_aol = [aol_func_map['and'] for key in test_method]
            print 'get real_aol value error'
        #不同的标签与或组合代理函数，实现不同标签能力的互补，或者制约，并且或逻辑可以增加逻辑结果为True连续性
        def advice_func_special(temp,and_or_logic_list=real_aol):
            return advice_func_general(temp, and_or_logic_list)
        code_perform = whole_simulation(codata,combination_func,advice_func_special,review_num,review_end,accerating=False)
        perform.append(code_perform)
        if code_perform!=0:
            not_zero_profit_dic[code]=code_perform
        #perform.append(whole_simulation(codata,combine4_simulation,advice_func_combine4,55))
        #print '                         code',code,i-1,'/',len_codes,'simulation'
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
    print '同期指数收益--',review_num,'---',review_end,':',
    for index_dic_key,index_dic_value in dtcst.INDEX_LIST.items():
        print index_dic_value,':',getlast_n_days_p_change(code=index_dic_key,n=review_num_cp,end=review_end_cp),
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
        return 0,[],[],0
    try:
        dislike_down_mean = sum(down_values_list)/len(down_values_list)
        dislike_up_mean = sum(up_values_list)/len(up_values_list)
    except:
        dislike_down_mean = 0
        dislike_up_mean = 0
    dislike_codes=[];dislike_codes2=[];dislike_codes3=[];dislike_down_thresh=dislike_down_mean/2;dislike_up_thresh = dislike_up_mean*1.5
    for key,value in not_zero_profit_dic.items():
        if value<=dislike_down_mean:#取负均值以下的
            dislike_codes.append(key)
        elif 0>value>dislike_down_thresh:
            pass
            #dislike_codes2.append(key)
        elif dislike_down_thresh>=value>dislike_down_mean:#取负均值到负值门限的#dislike_down_thresh=dislike_down_mean/2
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
    return profit_loss,[dislike_codes3,dislike_codes2,dislike_codes],down_value_codes,mean_target
#根据所给定周期参数，simu先用get_optimized_func_of_codes获取最有func 接着在回测天数下，
#进行whole_stock_ml_perform_simulation_of_func_code的总的perform计算测试
def simulation(review_num_train=250,interval_train = 20,interval_Loopback_Testing = 20,codes=None):
    #interval_train大于等于interval_Loopback_Testing
    review_num_start=review_num_train
    if codes==None:
        codes=dtcst.Whole_codes
    print codes
    train_num_list = util.type_data(range(review_num_train,interval_train,-interval_Loopback_Testing))
    train_profit_loss = []
    for review_num_train in train_num_list:
        review_end_train = review_num_train-interval_train;
        code_func_map ,func_name_address= get_optimized_func_of_codes(review_num_train, review_end_train,codes=codes)
        review_num=review_end_train;review_end = review_num-interval_Loopback_Testing;
        if review_end<0:
            review_end=0
        print 'review_num_train',review_num_train
        review_num_train
        profit_loss,_,_,_ = whole_stock_ml_perform_simulation_of_func_code(review_num,review_end,codes=codes,code_func_map=code_func_map,func_name_address=func_name_address)
        train_profit_loss.append(profit_loss)
    print '\n',train_profit_loss,'\n最终总的盈利情况: （各时间回测值之和）: ',sum(train_profit_loss)
    print '同期指数收益--',review_num_start-interval_train ,'----',0,'period:'
    for index_dic_key,index_dic_value in dtcst.INDEX_LIST.items():
        print index_dic_value,':',getlast_n_days_p_change(code=index_dic_key,n=review_num_start-interval_train ,end=0),
    return sum(train_profit_loss)
def get_optimized_period_review_num_of_simulation(whole_period=250,maybe_params=[],identity_perform=-999,codes=dtcst.Whole_codes):
    wanted_param=None;num_list=maybe_params;period_perform=[]
    for period in num_list:
        print 'period',period;interval_train=period;interval_Loopback_Testing = period
        simu_perform =simulation(review_num_train=whole_period+interval_train,interval_train=interval_train ,interval_Loopback_Testing=interval_Loopback_Testing,codes=codes)
        if simu_perform>=identity_perform:
            identity_perform=simu_perform;
            wanted_param=period
        period_perform.append(simu_perform)
        print 'period',period,'perform:',simu_perform
    print period_perform,'optimized param:',wanted_param
    return wanted_param
def ml_funcs_perform_compare():
        #全体方法测试排名
    import ml_all
    all_methods = ml_all.func_name_address.keys()
    all_methods = ['temp_test'];
    codes = dtcst.Whole_codes;
    for test_method in all_methods:
        whole_stock_ml_perform_simulation(review_num=255,review_end=0,codes=codes,not_want_codes_list=[],test_method=test_method)
 
#test_main.py 中控制台直接运行的代码
def test_main():
    get_optimized_period_review_num_of_simulation(maybe_params=[30,25,20,17,15,13,10,7,5])
    #simulation()
# if __name__ == '__main__':
#     review_num=175;review_end=0;codes=dtcst.Whole_codes;not_want_codes_list=[]
#     #codes=['603032']
#     if len(codes)==0:
#         print 'entering whole_stock_ml_perform_simulation: ',review_num,review_end,'not_wanted_codes_list:',len(not_want_codes_list),\
#         ':',not_want_codes_list,'\nwanted_codes:',len(codes)
#     perform  = [];not_zero_profit_dic={}
#     i=1;#codes = list(set(codes).difference(set((util.myflatten(not_want_codes_list)))))
#     print 'entering whole_stock_ml_perform_simulation: ',review_num,review_end,'not_wanted_codes_list:',len(not_want_codes_list),\
#         ':',not_want_codes_list,'\nwanted_codes:',len(codes)
#     len_codes = len(codes)
#     #使得大盘日期与个股日期回测日期映射保持日期一致性
#     review_num_date,review_end_date  = util.get_coherence_date(review_num, review_end)
#     review_num_cp = review_num;review_end_cp = review_end
#     for code in codes:
#         review_num = review_num_cp;review_end = review_end_cp;
#         print i,'/',len_codes,'simulation',code;i+=1;
#         try:
#             codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
#         except IOError,e:
#             print e;i-=1;len_codes-=1
#             continue
#         if len(codata)<=review_num+55:
#             print 'data amount too small'
#             i-=1;len_codes-=1
#             continue
#         #对review_num 和 review_end 重新映射使得满足和大盘日期一致性
#         review_num,review_num_date_or_wanted= util.review_num_end_mapping(codata,review_num_date,review_end_date,which_want_or_change_direction=1)
#         review_end,review_end_date= util.review_num_end_mapping(codata,review_end_date,review_num_date_or_wanted,which_want_or_change_direction=-1)
#         if review_num==None or review_end==None:continue
#         review_num+=1
#         from ml_stock import ml_all
#         code_perform = whole_simulation(codata,ml_all.get_optimized_func_of_codes_by_sequence_similarity,advice_func_general,review_num,review_end,accerating=False)
#         perform.append(code_perform)
#     print perform,'\n',sum(perform)/len(perform)

if __name__ == "__main__":
    whole_stock_ml_perform_simulation(review_num=255,review_end=0,codes=dtcst.Whole_codes,not_want_codes_list=[],test_method='default')

