# -*- coding:utf-8 -*-
#!/usr/bin/env python
'''计算指标接口'''
import pandas as pd
import numpy as np
import copy
'''const 常量定义'''
Derivative_str='_d';Time_Related_Index=['volume','amount','turnoverratio']
def get_daily_compute_index(last_second_data,code_rt_dt):
    temp = []; p_c=code_rt_dt['changepercent'].item();old_p_c=last_second_data['p_change'].item()
    volume =code_rt_dt['volume'].item()/100
    minimum=-1;maximum=1;normal=0;#result 存放驻点信息 极小值-1，极大值1，正常过渡点0
    #振幅amplitude计算
    amplitude = (code_rt_dt['high']-code_rt_dt['low'])/code_rt_dt['trade']*100
    temp.append(amplitude)
    #PVT 计算
    PVT=p_c*volume+last_second_data['PVT']
    temp.append(PVT)
    #stationary_info 今日设置为默认值，因为这个值的确定需要后一日的数据
    temp.append(normal)
    #昨日 stationary_info 设置，今日信息已得可以确定昨日驻点状态
    stationary_old=0
    if p_c>0:
        if old_p_c<0:stationary_old=minimum
    elif p_c<0:
        if old_p_c>0:stationary_old=maximum
    else :
        if old_p_c<0:stationary_old=minimum
        elif old_p_c>0:stationary_old=maximum
        else :stationary_old=normal
    #昨日信息可以更新的全部更新
    last_second_data = pd.Series(last_second_data);
    last_second_data['stationary_info']=stationary_old;last_second_data['review_pc']=p_c;
    #计算dist2last_min，dist2last_max
    dist2last_min=dist2last_max=1;
    if stationary_old==1:
            dist2last_min=last_second_data['dist2last_min']+1;
    elif stationary_old==-1:
        dist2last_max=last_second_data['dist2last_max']+1;
    else: 
        dist2last_max=last_second_data['dist2last_max']+1;
        dist2last_min=last_second_data['dist2last_min']+1;
    temp.append(dist2last_min);temp.append(dist2last_max);
    return temp,last_second_data
    
    
def new_data_line(last_exchange_day,last_second_data,code_rt_dt):
    third_party_data_index = ['open','high','trade','low','volume','nmc','amount','turnoverratio','changepercent','changepercent']#第二次changepercent指代review_pc的初值
    temp = [];temp.append(last_exchange_day)
    import util;time_coefficient = util.time_coefficient(last_exchange_day)
    for item in third_party_data_index:
        append = code_rt_dt[item].item()
        if item in Time_Related_Index:
            if item=='turnoverratio':item='turnover_rate';#
            append=append + (1-time_coefficient)*last_second_data[item].item()
        if item=='volume':append = append/100
        temp =temp +[append]
    daily_computing_items,last_second_data = get_daily_compute_index(last_second_data,code_rt_dt)
    temp +=daily_computing_items
    return temp,last_second_data
def add_wanted_index(data,item,basic_env=None):
    #import pandas as pd
    if item == 'turnover_rate':
        return add_turnover_rate(data, basic_env)
    elif item == 'p_change':
        return add_p_change(data)
    elif item == 'MACD' or item=='DIFF' or item=='DEM':
        result=macd(data)
        columns =['MACD','DIFF','DEM','ema_quick','ema_slow']
        df = pd.DataFrame({'MACD':result[0],'DIFF':result[1],'DEM':result[2],'ema_quick':result[3],'ema_slow':result[4]},columns =columns)
        df.index=data.index
        '''data = pd.concat([data,df], axis=1)'''#版本兼容性不好
        for item in columns:
            try:
                data.insert(6,item,df[item])
            except:print(item,'index data already existed')
        return data
    elif item== 'K' or item=='D' or item=='J' or item=='RSV':
        result=kdj(data)
        columns =['K','D','J','RSV']
        df=pd.DataFrame({'K':result[0],'D':result[1],'J':result[2],'RSV':result[3]},columns =columns)
        df.index=data.index
        '''data = pd.concat([data,df], axis=1)'''#版本兼容性不好
        for item in columns:
            try:
                data.insert(6,item,df[item])
            except:print(item,'index data already existed')
        return data
    elif item=='amount':
        return add_amount(data)
    elif item=='amplitude':
        return add_amplitude(data)
    elif item=='review_pc':
        return add_review_pc(data)
    elif item=='nmc':
        return add_nmc(data,basic_env)
    elif item=='MA5':
        result=ma(data,n=5)#默认10日均线
        df = pd.DataFrame({item:result},columns =[item])
        df.index=data.index
        '''data = pd.concat([data,df], axis=1)'''#版本兼容性不好
        data.insert(6,item,df[item])
        return data
    elif item=='ema3' or item=='ema5':
        n=int(item[-1])
        result=ema(data,n=n)
        df = pd.DataFrame({item:result},columns =[item])
        df.index=data.index
        data.insert(6,item,df[item])
        
        data=add_p_change(data, input_s=item+'_')
        data=add_PVT(data, input_s=item+'_')
        data=add_stationary_info(data, input_s=item+'_')
        return data
    elif item=='efficiency_coefficient':
        return add_efficiency_coefficient(data)
    elif item=='backward_inference_MACD_hit':
        return add_backward_inference_MACD_hit(data)
    elif item=='is_break_up_n_high_or_low':
        return add_is_break_up_n_high_or_low(data)
    elif item=='number_stationary_in_n_days':
        return add_number_stationary_in_n_days(data)
    elif item=='is_d_bigger_than_d_mean':
        return add_is_d_bigger_than_d_mean(data)
    elif item=='continuous_up_down_day':
        return add_continuous_up_down_day(data)
    elif item=='big_drop_rise_review_days':
        return add_big_drop_rise_review_days(data)
    elif item=='volume_pc_feature':
        return add_volume_pc_feature(data)
    elif item=='recommend_period':
        return add_recommend_period(data)
    elif item == 'fear_capital_feature':
        return add_fear_capital_feature(data)
    elif item =='greedy_capital_feature':
        return add_greedy_capital_feature(data)
    elif item == 'me_tooism_human_feature':
        # 内部计算了 fear_human_feature
        return add_me_tooism_human_feature(data)
    elif item == 'feeble_feature':
        return add_feeble_feature(data)
    elif item == 'close_ama':
        data = add_close_ama(data);
        
        data=add_p_change(data, input_s=item+'_');
        data=add_PVT(data, input_s=item+'_');
        data=add_stationary_info(data, input_s=item+'_');
        return data
    elif item=='positive_volume_sum':
        return add_positive_volume_sum(data)
    elif item=='PVT':
        return add_PVT(data)
    elif item=='stationary_info':
        return add_stationary_info(data)
    elif item=="dist2last_big_drop":
        return add_dist2last_big_drop(data)
    elif item=='pseudo_mean_price':
        data = add_psedo_mean_price(data);
        
        data=add_p_change(data, input_s=item+'_');
        data=add_PVT(data, input_s=item+'_');
        data=add_stationary_info(data, input_s=item+'_');
        return data
    else:
        print 'this is a index not in const pool:',item ,' .'
        raise
        return data
def add_psedo_mean_price(data):
    label='pseudo_mean_price' 
    psedo_candidate=data[['close','high','low']]
    data.insert(6,label,psedo_candidate.mean(axis = 1)) ;
    return data
def add_PVT(data,input_s=''):#还包括计算 VT 纯volume trend 指标
    volume='volume';p_change=input_s+'p_change';PVT=input_s+'PVT';VT=input_s+'VT';
    p_changes=data[p_change].values;volumes=data[volume].values;
    d_pvts = volumes * p_changes;   sign = map(lambda x:1 if x>0 else -1, data[p_change].values);
    vts=np.array(sign)*volumes; 
    for i in range(len(d_pvts))[1:]:#跳过第一项因为第一项的前一项为0，所以加和还是本身
        d_pvts[i]+=d_pvts[i-1];  vts[i]+=vts[i-1];
    result = d_pvts
    data.insert(6,PVT,result);  data.insert(6,VT,vts);
    return data
def add_positive_volume_sum(data,_index='p_change'):
    list_my = data[_index].tolist();volume = data['volume'].tolist()
    positive_v_continuous_level=[0]*len(volume);negative_v_continuous_level=[0]*len(volume)#正池水平，负池水平
    positive_v_continuous_portion_nega_sum=[0]*len(volume);negative_v_continuous_portion_posi_sum=[0]*len(volume); #已正/负sum 已负/正sum
    positive_volume_sum=[volume[0]]*len(volume);negative_volume_sum=[volume[0]]*len(volume)
    for i in range(len(list_my)):
        ii=i-1;p_v_c_sum=0;n_v_c_sum=0;
        if list_my[i]>0 and list_my[ii]>0:
            positive_volume_sum[i]=positive_volume_sum[ii];negative_volume_sum[i]=negative_volume_sum[ii]
            for iii in range(i,-1,-1):
                if list_my[iii]>0:p_v_c_sum+=volume[iii]*abs(list_my[iii])
                else :break
        elif list_my[i]<0 and list_my[ii]<0:
            positive_volume_sum[i]=positive_volume_sum[ii];negative_volume_sum[i]=negative_volume_sum[ii]
            for iii in range(i,-1,-1):
                if list_my[iii]<0:n_v_c_sum+=volume[iii]*abs(list_my[iii])
                else :break
        elif list_my[i]*list_my[ii]<0:#今日和昨日相乘异号，状态转换
            prior_v_sum=0;
            if list_my[i]<0:  #今日状态小于0，更新今日n_v_c_sum,更新今日以前求和项p_v_sum
                n_v_c_sum=volume[i]*abs(list_my[i])
                for iii in range(i-1,-1,-1):
                    if list_my[iii]>0:prior_v_sum+=volume[iii]*abs(list_my[iii])
                    else:break
                if prior_v_sum!=0:positive_volume_sum[i]=prior_v_sum*abs(list_my[i]);
                negative_volume_sum[i]=negative_volume_sum[ii]
            elif list_my[i]>0:
                p_v_c_sum=volume[i]*abs(list_my[i])
                for iii in range(i-1,-1,-1):
                    if list_my[iii]<0:prior_v_sum+=volume[iii]*abs(list_my[iii])
                    else:break
                if prior_v_sum!=0:negative_volume_sum[i]=prior_v_sum
                positive_volume_sum[i]=positive_volume_sum[ii];
        thresh_posi = 4;thresh_nega=thresh_posi;
        if list_my[i]>=thresh_posi or float(p_v_c_sum)/positive_volume_sum[i]>1:positive_volume_sum[i]+=volume[i];
        if list_my[i]<=thresh_nega or float(n_v_c_sum)/negative_volume_sum[i]>1:negative_volume_sum[i]+=volume[i]
        positive_v_continuous_level[i]=float(p_v_c_sum)/positive_volume_sum[i];
        negative_v_continuous_level[i] =float(n_v_c_sum)/negative_volume_sum[i];
        positive_v_continuous_portion_nega_sum[i] =float(p_v_c_sum)/negative_volume_sum[i];
        negative_v_continuous_portion_posi_sum[i] =float(n_v_c_sum)/positive_volume_sum[i];
        #print list_my[ii],list_my[i],volume[i],'p_v_c_sum',p_v_c_sum,'positive_volume_sum',positive_volume_sum[i],positive_v_continuous_level[i]
    data.insert(6,'positive_volume_sum',positive_volume_sum)
    data.insert(6,'positive_v_continuous_level',positive_v_continuous_level)
    data.insert(6,'negative_volume_sum',negative_volume_sum)
    data.insert(6,'negative_v_continuous_level',negative_v_continuous_level)
    data.insert(6,'positive_v_continuous_portion_nega_sum',positive_v_continuous_portion_nega_sum)
    data.insert(6,'negative_v_continuous_portion_posi_sum',negative_v_continuous_portion_posi_sum)
    return data
def add_close_ama(data):
    '''direction = price - price[len];
    价格方向：len个时间周期中价格的净变化
    波动性，市场噪音的数量，计算时使用len个时间周期中所有单周期价格变化的总和。
    volatility = @sum(@abs(price –price[1]), n);
    效率系数：价格方向除以波动性，表示方向移动与噪音移动的比。
    Efficiency_Ratio =direction/volativity;'''
    #需要注意效率系数的计算周期 默认n=10
    efficiency_coefficient = data['efficiency_coefficient'].values;
    close = data['close'].values;
    PRICE=0; n_f=2;n_s=26
    fastest = 2/(n_f+1.000);
    slowest = 2/(n_s+1.000);
    smooth = efficiency_coefficient*(fastest - slowest)+ slowest;
    C = smooth*smooth;
    AMA=[close[0]]*len(close);#初值使用技巧，利用只有一个元素时-1 指向0-index 本身，for循环中可以避免初值if判断
    for index in range(len(close)):
        PRICE=close[index];c=C[index]
        AMA[index]=(AMA[index-1]+c*(PRICE-AMA[index-1]) )
        
    result = AMA
    data.insert(6,'close_ama',result)
    return data
    '''为了与系统自适应特性保持一致，不能简单的用上穿下穿均线来决定买入卖出。因此要设置一个过滤器。
    过滤器=percentage*@std（AMA-AMA[1],n）          @std（series，n）是n个周期标准差
    小的过滤器百分数可以用于较快的交易，比如外汇与期货市场。
    大的过滤器百分数可以用于较慢的交易，比如股票和利率市场。
    通常，n=20
    具体交易规则：
    AMA-@lowest(AMA,n)>过滤器，买入
    @highest（AMA，n）-AMA<过滤器，卖出''' 
def add_feeble_feature(data):
    import numpy as np
    continuous_up_down_day = data['continuous_up_down_day'].tolist();vl_pc =data['volume_pc_feature'].tolist()
    #recommend_period =data['recommend_period'].tolist(); 
    old_today =int(-5);#-recommend_period[-1]
    now_c_u_d_d = continuous_up_down_day[-1]
    temp_list =continuous_up_down_day[old_today :-1];positive_list=[];negative_list=[]
    for ii in temp_list:
                if ii >0:positive_list.append(ii)
                elif ii<0:negative_list.append(ii)
    positive_mean = np.mean(positive_list);negative_mean=np.mean(negative_list);
    #portion = positive_mean/(now_c_u_d_d+0.000)
    #portion = negative_mean/(now_c_u_d_d+0.000)
    #result = [portion for ii in continuous_up_down_day]
    result = continuous_up_down_day
    data.insert(6,'feeble_feature',result)
    return data
def add_me_tooism_human_feature(data):
    p_c =data['p_change'].values;continuous_up_down_day = data['continuous_up_down_day'].values
    me_tooism=p_c*continuous_up_down_day
    
    result = me_tooism
    data.insert(6,'me_tooism_human_feature',result)
    
    fear_human_feature=-me_tooism;result=fear_human_feature;
    data.insert(6,'fear_human_feature',result)
    return data
def add_greedy_capital_feature(data):
    fear_capital_feature = data['fear_capital_feature'];
    greedy=1-fear_capital_feature
    
    result=greedy
    data.insert(6,'greedy_capital_feature',result)
    return data
def add_fear_capital_feature(data):
    e_c = data['efficiency_coefficient'].values;volume_pc = data['volume_pc_feature'];quantile_close_in_period=0
    fear_capital_feature=-(e_c+volume_pc)+quantile_close_in_period
    
    result=fear_capital_feature
    data.insert(6,'fear_capital_feature',result)
    return data
def add_recommend_period(data,n=15):
    import numpy as np
    all_periods=data['big_drop_rise_review_days'].values;n_stationary_period = data['number_stationary_in_n_days'].values
    T = (n+0.000)/5*2 #T=long/(stationary_n/2) n_stationary_period
    
    result = T
    data.insert(6,'recommend_period',result)
    return data
def add_volume_pc_feature(data):
    import numpy as np
    p_c=data['p_change'].values;volume_ = data['volume'].values
    union_feature = p_c*volume_
    def interval_mapping(old_list,max_w=1.00000,min_w=-1.0000):
        value=old_list;interval_ = max_w-min_w;
        max_v=max(value);min_v=min(value)
        k = interval_/(max_v-min_v+0.0000)
        transform_value=[k*(x-min_v)+min_w for x in value]
        return transform_value
    result = interval_mapping(old_list=union_feature, max_w=1, min_w=-1)
    data.insert(6,'volume_pc_feature',result)
    return data
def add_big_drop_rise_review_days(data,n_period=15):
    _index='p_change';import numpy as np
    list_my = data[_index].tolist();result=[];temp_list=[]
    for i in range(len(list_my)):
        temp=n_period
        if i <n_period:temp=0 if np.mean(list_my[:i+1])>=3 else i
        else :
            temp_list=list_my[:i+1]
            positive_list = [];negative_list=[]; 
            for ii in temp_list:
                if ii >0:positive_list.append(ii)
                elif ii<0:negative_list.append(ii)
            if np.mean([abs(ii) for ii in temp_list])>5*np.mean([abs(ii)for ii in list_my[:i+1]]):temp=0
            else:
                positive_mean=np.mean(positive_list);negative_mean=np.mean(negative_list);
                for last_insane in range(len(temp_list)-2,n_period,-1): #last_insane最后的疯狂点
                    abs_15_index_mean=np.mean([abs(ii)for ii in temp_list[last_insane-n_period:last_insane]])
                    if abs_15_index_mean>3*positive_mean or abs_15_index_mean<5*negative_mean:
                        temp=i-last_insane;break
                    #print abs_15_index_mean,positive_mean,negative_mean
        result.append(temp)
    data.insert(6,'big_drop_rise_review_days',result)
    return data

def add_continuous_up_down_day(data,_index='p_change'):
    list_my = data[_index].tolist();result=[];temp_list=[]
    for i in range(len(list_my)):
        temp=0;ii=i;
        while ii>0:
            ii-=1;
            if list_my[i]>0 and list_my[ii]>0:temp+=1
            elif list_my[i]<0 and list_my[ii]<0:temp-=1
            else: break
        result.append(temp)
    data.insert(6,'continuous_up_down_day',result)
    return data
def add_is_d_bigger_than_d_mean(data,_index='MACD_d',n=15):
    import numpy
    list_my = data[_index].tolist();result=[];temp_list=[]
    for i in range(len(list_my)):
        if i<n:temp_list = list_my[:i+1]
        else:temp_list = list_my[i+1-n:i+1]
        temp=list_my[i];positive_list = [];negative_list=[]; 
        for ii in temp_list:
            if ii >0:positive_list.append(ii)
            elif ii<0:negative_list.append(ii)
        if temp>=numpy.mean(positive_list):temp=1;
        elif temp<=numpy.mean(negative_list):temp=-1;
        else :temp=0;
        result.append(temp)
    data.insert(6,'is_d_bigger_than_d_mean',result)
    return data
def add_stationary_info(data,input_s=''):
    minimum=-1;maximum=1;normal=0;#result 存放驻点信息 极小值-1，极大值1，正常过渡点0
    p_change=input_s+'p_change';stationary_info=input_s+'stationary_info';
    dist2last_min=input_s+'dist2last_min';dist2last_max=input_s+'dist2last_max'
    list_my = data[p_change].tolist();result=[normal]*len(list_my);
    for i in range(1,len(list_my)):#每个点的驻点属性只能由下一个点来确定
        if list_my[i]>0:
            if list_my[i-1]<0:result[i-1]=minimum
        elif list_my[i]<0:
            if list_my[i-1]>0:result[i-1]=maximum
        else :
            if list_my[i-1]<0:result[i-1]=minimum
            elif list_my[i-1]>0:result[i-1]=maximum
            else :result[i-1]=normal
    data.insert(6,stationary_info,result)
    dist2last_min_temp=[1]*len(list_my);dist2last_max_temp=[1]*len(list_my)
    min_reset=0;max_reset=0;
    for i in range(1,len(list_my)):#每个点的驻点属性只能由下一个点来确定
        if max_reset==1:
            dist2last_min_temp[i]=dist2last_min_temp[i-1]+1;max_reset=0
        elif min_reset==1:
            dist2last_max_temp[i]=dist2last_max_temp[i-1]+1;min_reset=0
        else: 
            dist2last_max_temp[i]=dist2last_max_temp[i-1]+1;
            dist2last_min_temp[i]=dist2last_min_temp[i-1]+1
        if result[i]>0:max_reset=1
        elif result[i]<0:min_reset=1
        else :pass
    data.insert(6,dist2last_min,dist2last_min_temp);data.insert(6,dist2last_max,dist2last_max_temp)
    return data
def add_dist2last_big_drop(data,input_s="p_change"):
    """
    计算到各个暴跌点的距离，暴跌点是新生的时间，股票新的交易模式的开启
    判断今天是否是暴跌点，是的话 填dist2last_big_drop 为0
    否则将昨日的dist2last_big_drop 加一作为今日的。
    :param data:
    :param input_s:
    :return:
    """
    dist2last_big_drop_init = 0  #新的一轮开启
    p_change_data = data[input_s].values
    volume_data   = data["volume"].values
    close_data    = data["close"].values
    result = copy.deepcopy(p_change_data)
    last_n = 5
    drop_limit = -4
    volume_now_days = 15
    volume_previdous_days = int(np.e*volume_now_days)
    volume_limit_propotion = np.e
    for i in range(0,len(p_change_data)):#每个点的驻点属性只能由下一个点来确定
        #用涨跌幅判断是否大跌
        if i<last_n:
            be_mean_data = p_change_data[:i+1]
        else:
            be_mean_data = p_change_data[i-last_n:i+1]
        mean_drop = np.mean(be_mean_data)

        #用成交量的缩水判断是否大跌
        if i < volume_now_days + volume_previdous_days+1:
            volume_propotion = 1
        else:
            volume_propotion = np.mean(volume_data[i-(volume_now_days+volume_previdous_days):i-volume_now_days]) \
                               / np.mean(volume_data[i-volume_now_days:i+1])

        #用是否是近期低点判断并且最近几日跌幅较大，避免振荡后的触底判断成大跌
        if np.min(close_data[:i+1][-55:]) == close_data[i] and np.mean(p_change_data[:i+1][-3:])<drop_limit:
            recent_min_judge_is_big_drop = True
        else:
            recent_min_judge_is_big_drop = False

        # 综上所述
        if mean_drop <= drop_limit or i == 0 or volume_limit_propotion < volume_propotion or recent_min_judge_is_big_drop:
            result[i] = dist2last_big_drop_init
        else:
            result[i] = result[i-1]+1
    result = result.astype(np.int)
    data.insert(6,"dist2last_big_drop",result)
    return data
def add_number_stationary_in_n_days(data,n=15):
    list_my = data['p_change'].tolist();result=[]
    for i in range(len(list_my)):
        if i<n:temp_list = list_my[:i+1]
        else:temp_list = list_my[i+1-n:i+1]
        temp=0
        if len(temp_list)>1:
            for ii in range(1,len(temp_list)):
                if temp_list[ii]*temp_list[ii-1]<0:temp+=1 #驻点
        result.append(temp)
    data.insert(6,'number_stationary_in_n_days',result)
    return data
def add_is_break_up_n_high_or_low(data,n=10):
    #print 'entering add_is_break_up_n_high_or_low computing', 
    list_my = data['close'].tolist();result=[];max_n=0;min_n=0;
    for i in range(len(list_my)):
        temp=0;
        if i<n:
            max_n=max(list_my[:i+1]);min_n=min(list_my[:i+1]);
        else:
            max_n = max(list_my[i+1-n:i+1]);min_n=min(list_my[i+1-n:i+1]);
        if list_my[i]>=max_n:temp=1;
        elif list_my[i]<=min_n:temp = -1
        result.append(temp)
    data.insert(6,'is_break_up_n_high_or_low',result)
    return data
def add_backward_inference_MACD_hit(data):
    def continuous_MACD_compute(close,last_second_data):
        data=last_second_data;quick_n=12;slow_n=26;dem_n=9;ema_map={'_quick':quick_n,'_slow':slow_n,'5':5}
        for ema_key in ema_map:
            ema_map[ema_key] = (2 * close + ( ema_map[ema_key] - 1) * data['ema'+ema_key]) / ( ema_map[ema_key] + 1)
        ema_quick = ema_map['_quick']
        ema_slow = ema_map['_slow']
        DIFF = ema_quick -ema_slow
        DEM = (2 * DIFF + (dem_n - 1) * data['DEM']) / (dem_n + 1)
        MACD = 2*(DIFF-DEM);return MACD;
    #last_second_data = data.iloc[-2];last_data=data.iloc[-1];high = last_data['high'];low = last_data['low']
    def hitter_judger(last_second_data,high,low):
        MACD_d_old_bigger_0 = last_second_data['MACD_d']>0;high_MACD_compare_old = continuous_MACD_compute(high, last_second_data)-last_second_data['MACD']
        low_MACD_compare_old = continuous_MACD_compute(low, last_second_data)-last_second_data['MACD']
        if MACD_d_old_bigger_0 and low_MACD_compare_old<0:return -1;
        elif not MACD_d_old_bigger_0 and high_MACD_compare_old >0:return 1;
        else: return 0;
    result =[0]#第一位置零
    for ii in range(1,len(data)):
        last_data=data.iloc[ii];last_second_data=data.iloc[ii-1];high = last_data['high'];low = last_data['low']
        result.append(hitter_judger(last_second_data, high, low))
    data.insert(6,'backward_inference_MACD_hit',result)
    return data
def add_efficiency_coefficient(data,n=10):
    #print 'entering efficiency_coefficient computing', 
    list_my = data['p_change'].tolist();result=[]
    old=0;new=0;whole_sum=0;whole_abs_sum=0;
    for i in range(len(list_my)):
        if i<n:
            whole_sum = sum(list_my[:i+1]);whole_abs_sum = sum([abs(ii) for ii in list_my[:i+1]]);
            old = list_my[0]
        else:
            new = list_my[i]
            whole_sum = whole_sum+new-old;whole_abs_sum +=(-abs(old)+abs(new));
            old = list_my[i-n+1]
        if whole_abs_sum==0:temp=0;
        else:temp = whole_sum/whole_abs_sum;
        result.append(temp)
    data.insert(6,'efficiency_coefficient',result)
    return data
            
def add_turnover_rate(data_copy,basic_env):
    outstanding = basic_env['outstanding'] #单位亿股 10**8
    volume = data_copy['volume'] #成交量单位手----100股
    unit_multier = 100.0000/10000/10000*100
    turnover_rate = volume/outstanding*unit_multier #换手率等于成交量除以流通股本,转换为百分数需再*100
    data_copy.insert(6,'turnover_rate',turnover_rate)
    return data_copy
def add_amount(data_copy):
    volume = data_copy['volume'] #成交量单位手----100股
    close = data_copy['close']
    unit_multier = 100
    amount = volume*close*unit_multier #换手率等于成交量除以流通股本,转换为百分数需再*100
    data_copy.insert(6,'amount',amount)
    return data_copy
def add_amplitude(data):
    high = data['high'] #成交量单位手----100股
    close = data['close']
    low = data['low']
    unit_multier = 100
    amplitude = (high-low)/close*unit_multier #换手率等于成交量除以流通股本,转换为百分数需再*100
    data.insert(6,'amplitude',amplitude)
    return data
def add_review_pc(data):
    p_change = data['p_change']
    review_pc = p_change.shift(-1)
    review_pc.loc[review_pc.index[-1]]=p_change.loc[p_change.index[-1]]
    data.insert(6,'review_pc',review_pc)
    return data
def add_nmc(data,basic_env):
    nmc  = basic_env['outstanding'].item() * data.tail(1)['close'].item()*10000#nmc单位万元 #outstanding 流通股本数量单位亿股 10**8
    nmc_series = pd.Series(nmc, index=data.index.tolist())
    data.insert(6,'nmc',nmc_series)
    return data
def add_p_change(data,input_s='close'):
    close=input_s[:-1] if input_s!='close' else 'close' 
    p_change='p_change' if close=='close' else input_s+'p_change';
    a=add_relative_change(data,close)
    a.rename(columns={close+'_d':p_change}, inplace = True)
    a.loc[:,p_change]=a.loc[:,p_change]*100
    return a
def difference(data,index):
    y = data[index];
    old_y=y.shift(1)
    dy=y.sub(old_y,fill_value=0)#old_y第一个数字居然是nan
    dy[dy.index[0]]=0
    return dy
def add_relative_change(data,index,relative_obj=None,n=None):
    old_y=data[index].shift(1);old_y[old_y.index[0]]=old_y[old_y.index[1]]#数据往index 加1方向挪old_y
    if 0 in old_y.values:raise# 否则判断索引是否有某个元素
    dy=difference(data, index)
    r_c=dy.div(old_y.abs(),fill_value=0)
    '''for i in range(len(dy)):
        if i<5:print i,':',data[index][i],'old:',old_y[i],'diff:',dy[i]
        elif i>len(dy)-5:print i,'old:',old_y[i],'diff:',dy[i]
    raise'''
    r_c[r_c.index[0]]=r_c[r_c.index[1]];
    data.insert(1,index + '_d',r_c)
    return data
'''
添加指标的差分
'''
def add_index_derivative(data,index):
    dy = difference(data, index)
    data.insert(1,index + '_d',dy)
    return data
def ma(data, n=10, val_name="close"):
    import numpy as np
    '''
    移动平均线 Moving Average
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
                  移动平均线时长，时间单位根据data决定
      val_name:string
                  计算哪一列的列名，默认为 close 收盘值

    return
    -------
      list_my
          移动平均线
    '''
    values = []
    MA = []
    for index, row in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]
        MA.append(np.average(values))
    return np.asarray(MA)
def md(data, n=10, val_name="close"):
    import numpy as np
    '''
    移动标准差
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
                  移动平均线时长，时间单位根据data决定
      val_name:string
                  计算哪一列的列名，默认为 close 收盘值
    return
    -------
      list_my
          移动平均线
    '''
    values = []
    MD = []

    for index, row in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]

        MD.append(np.std(values))

    return np.asarray(MD)
def _get_day_ema(prices, n):
    a = 1 - 2 / (n + 1)

    day_ema = 0
    for index, price in enumerate(reversed(prices)):
        day_ema += a ** index * price

    return day_ema
def ema(data, n=12, val_name="close"):
    import numpy as np
    '''
        指数平均数指标 Exponential Moving Average
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                      移动平均线时长，时间单位根据data决定
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值
        return
        -------
          EMA:numpy.ndarray<numpy.float64>
              指数平均数指标
    '''
    prices = []
    EMA = []
    index0=data.index[0]
    for index, row in data.iterrows():
        if index == index0:
            past_ema = row[val_name]
            EMA.append(row[val_name])
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            today_ema = (2 * row[val_name] + (n - 1) * past_ema) / (n + 1)
            past_ema = today_ema
            EMA.append(today_ema)
    return np.asarray(EMA)
def macd(data, quick_n=12, slow_n=26, dem_n=9, val_name="close"):
    import numpy as np
    '''
        指数平滑异同平均线(MACD: Moving Average Convergence Divergence)
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          quick_n:int
                      DIFF差离值中快速移动天数
          slow_n:int
                      DIFF差离值中慢速移动天数
          dem_n:int
                      DEM讯号线的移动天数
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值
        return
        -------
          MACD:numpy.ndarray<numpy.float64>
              MACD bar / OSC 差值柱形图 DIFF - DEM 的两倍
          DIFF:numpy.ndarray<numpy.float64>
              差离值
          DEM:numpy.ndarray<numpy.float64>
              讯号线
    '''
    ema_quick = np.asarray(ema(data, quick_n, val_name))
    ema_slow = np.asarray(ema(data, slow_n, val_name))
    ema_slow[0]=ema_slow[0]*1.001
    DIFF = ema_quick - ema_slow
    data["diff"] = DIFF
    DEM = ema(data, dem_n, "diff")
    MACD = (DIFF - DEM)*2
    MACD[0] = MACD.mean()
    return MACD, DIFF, DEM,ema_quick,ema_slow
def kdj(data):
    import numpy as np
    '''
        随机指标KDJ
        Parameters
        ------
          data:pandas.DataFrame
                通过 get_k_data 取得的股票数据
        return
        -------
          K:numpy.ndarray<numpy.float64>
              K线
          D:numpy.ndarray<numpy.float64>
              D线
          J:numpy.ndarray<numpy.float64>
              J线
    '''

    K, D, J = [], [], [];RSV=[]
    last_k, last_d = None, None
    for index, row in data.iterrows():
        if last_k is None or last_d is None:
            last_k = 50
            last_d = 50
        c, l, h = row["close"], row["low"], row["high"]
        if h!=l:
            rsv = (c - l) / (h - l) * 100
        else:
            #print('h==l',last_k)
            if row[ 'p_change']>0:rsv=75
            else :rsv=25
        RSV.append(rsv/100.0)
        k = (2.0 / 3) * last_k + (1.0 / 3) * rsv
        d = (2.0 / 3) * last_d + (1.0 / 3) * k
        j = 3 * k - 2 * d
        K.append(k)
        D.append(d)
        J.append(j)
        last_k, last_d = k, d
    return np.asarray(K), np.asarray(D), np.asarray(J),np.asarray(RSV)


def rsi(data, n=6, val_name="close"):
    import numpy as np

    '''
        相对强弱指标RSI
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                统计时长，时间单位根据data决定
        return
        -------
          RSI:numpy.ndarray<numpy.float64>
              RSI线
        
    '''

    RSI = []
    UP = []
    DOWN = []
    index0=data.index[0]
    for index, row in data.iterrows():
        if index == index0:
            past_value = row[val_name]
            RSI.append(0)
        else:
            diff = row[val_name] - past_value
            if diff > 0:
                UP.append(diff)
                DOWN.append(0)
            else:
                UP.append(0)
                DOWN.append(diff)

            if len(UP) == n:
                del UP[0]
            if len(DOWN) == n:
                del DOWN[0]

            past_value = row[val_name]

            rsi = np.sum(UP) / (-np.sum(DOWN) + np.sum(UP)) * 100
            RSI.append(rsi)

    return np.asarray(RSI)


def boll(data, n=10, val_name="close", k=2):
    '''
        布林线指标BOLL
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                统计时长，时间单位根据data决定
        return
        -------
          BOLL:numpy.ndarray<numpy.float64>
              中轨线
          UPPER:numpy.ndarray<numpy.float64>
              D线
          J:numpy.ndarray<numpy.float64>
              J线
    '''

    BOLL = ma(data, n, val_name)

    MD = md(data, n, val_name)

    UPPER = BOLL + k * MD

    LOWER = BOLL - k * MD

    return BOLL, UPPER, LOWER


def wnr(data, n=14):
    '''
        威廉指标 w&r
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                统计时长，时间单位根据data决定
        return
        -------
          WNR:numpy.ndarray<numpy.float64>
              威廉指标
    '''

    high_prices = []
    low_prices = []
    WNR = []

    for index, row in data.iterrows():
        high_prices.append(row["high"])
        if len(high_prices) == n:
            del high_prices[0]
        low_prices.append(row["low"])
        if len(low_prices) == n:
            del low_prices[0]

        highest = max(high_prices)
        lowest = min(low_prices)

        wnr = (highest - row["close"]) / (highest - lowest) * 100
        WNR.append(wnr)

    return WNR


def _get_any_ma(arr, n):
    import numpy as np
    MA = []
    values = []
    for val in arr:
        values.append(val)
        if len(values) == n:
            del values[0]
        MA.append(np.average(values))
    return np.asarray(MA)


def dmi(data, n=14, m=14, k=6):
    import numpy as np

    '''
        动向指标或趋向指标 DMI
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              +-DI(n): DI统计时长，默认14
          m:int
              ADX(m): ADX统计时常参数，默认14
              
          k:int
              ADXR(k): ADXR统计k个周期前数据，默认6
        return
        -------
          P_DI:numpy.ndarray<numpy.float64>
              +DI指标
          M_DI:numpy.ndarray<numpy.float64>
              -DI指标
          ADX:numpy.ndarray<numpy.float64>
              ADX指标
          ADXR:numpy.ndarray<numpy.float64>
              ADXR指标
        ref.
        -------
        https://www.mk-mode.com/octopress/2012/03/03/03002038/
    '''

    # 上升动向（+DM）
    P_DM = [0.]
    # 下降动向（-DM）
    M_DM = [0.]
    # 真实波幅TR
    TR = [0.]
    # 动向
    DX = [0.]

    P_DI = [0.]
    M_DI = [0.]
    index0=data.index[0]
    for index, row in data.iterrows():
        if index == index0:
            past_row = row
        else:

            p_dm = row["high"] - past_row["high"]
            m_dm = past_row["low"] - row["low"]

            if (p_dm < 0 and m_dm < 0) or (np.isclose(p_dm, m_dm)):
                p_dm = 0
                m_dm = 0
            if p_dm > m_dm:
                m_dm = 0
            if m_dm > p_dm:
                p_dm = 0

            P_DM.append(p_dm)
            M_DM.append(m_dm)

            tr = max(row["high"] - past_row["low"], row["high"] - past_row["close"], past_row["close"] - row["low"])
            TR.append(tr)

            if len(P_DM) == n:
                del P_DM[0]
            if len(M_DM) == n:
                del M_DM[0]
            if len(TR) == n:
                del TR[0]

            # 上升方向线(+DI)
            p_di = (np.average(P_DM) / np.average(TR)) * 100
            P_DI.append(p_di)

            # 下降方向线(-DI)
            m_di = (np.average(M_DM) / np.average(TR)) * 100
            M_DI.append(m_di)

            # 当日+DI与-DI
            # p_day_di = (p_dm / tr) * 100
            # m_day_di = (m_dm / tr) * 100

            # 动向DX
            #     dx=(di dif÷di sum) ×100
            # 　　di dif为上升指标和下降指标的价差的绝对值
            # 　　di sum为上升指标和下降指标的总和
            # 　　adx就是dx的一定周期n的移动平均值。
            if (p_di + m_di) == 0:
                dx = 0
            else:
                dx = (abs(p_di - m_di) / (p_di + m_di)) * 100
            DX.append(dx)

            past_row = row

    ADX = _get_any_ma(DX, m)
    #
    # # 估计数值ADXR
    ADXR = []
    for index, adx in enumerate(ADX):
        if index >= k:
            adxr = (adx + ADX[index - k]) / 2
            ADXR.append(adxr)
        else:
            ADXR.append(0)

    return P_DI, M_DI, ADX, ADXR


def bias(data, n=5):
    import numpy as np
    '''
        乖离率 bias
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认5
        return
        -------
          BIAS:numpy.ndarray<numpy.float64>
              乖离率指标

    '''

    MA = ma(data, n)
    CLOSES = data["close"]
    BIAS = (np.true_divide((CLOSES - MA), MA)) * (100 / 100)
    return BIAS


def asi(data, n=5):
    import numpy as np
    '''
        振动升降指标 ASI
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认5
        return
        -------
          ASI:numpy.ndarray<numpy.float64>
              振动升降指标

    '''

    SI = []
    index0=data.index[0]
    for index, row in data.iterrows():
        if index == index0:
            last_row = row
            SI.append(0.)
        else:

            a = abs(row["close"] - last_row["close"])
            b = abs(row["low"] - last_row["close"])
            c = abs(row["high"] - last_row["close"])
            d = abs(last_row["close"] - last_row["open"])

            if b > a and b > c:
                r = b + (1 / 2) * a + (1 / 4) * d
            elif c > a and c > b:
                r = c + (1 / 4) * d
            else:
                r = 0

            e = row["close"] - last_row["close"]
            f = row["close"] - last_row["open"]
            g = last_row["close"] - last_row["open"]

            x = e + (1 / 2) * f + g
            k = max(a, b)
            l = 3

            if np.isclose(r, 0) or np.isclose(l, 0):
                si = 0
            else:
                si = 50 * (x / r) * (k / l)

            SI.append(si)

    ASI = _get_any_ma(SI, n)
    return ASI


def vr(data, n=26):
    import numpy as np
    '''
        Volatility Volume Ratio 成交量变异率
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认26
        return
        -------
          VR:numpy.ndarray<numpy.float64>
              成交量变异率

    '''
    VR = []

    AV_volumes, BV_volumes, CV_volumes = [], [], []
    for index, row in data.iterrows():

        if row["close"] > row["open"]:
            AV_volumes.append(row["volume"])
        elif row["close"] < row["open"]:
            BV_volumes.append(row["volume"])
        else:
            CV_volumes.append(row["volume"])

        if len(AV_volumes) == n:
            del AV_volumes[0]
        if len(BV_volumes) == n:
            del BV_volumes[0]
        if len(CV_volumes) == n:
            del CV_volumes[0]

        avs = sum(AV_volumes)
        bvs = sum(BV_volumes)
        cvs = sum(CV_volumes)

        if (bvs + (1 / 2) * cvs) != 0:
            vr = (avs + (1 / 2) * cvs) / (bvs + (1 / 2) * cvs)
        else:
            vr = 0

        VR.append(vr)

    return np.asarray(VR)


def arbr(data, n=26):
    import numpy as np

    '''
        AR 指标 BR指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认26
        return
        -------
          AR:numpy.ndarray<numpy.float64>
              AR指标
          BR:numpy.ndarray<numpy.float64>
              BR指标

    '''

    H, L, O, PC = np.array([0]), np.array([0]), np.array([0]), np.array([0])

    AR, BR = np.array([0]), np.array([0])
    index0=data.index[0]
    for index, row in data.iterrows():
        if index == index0:
            last_row = row

        else:

            h = row["high"]
            H = np.append(H, [h])
            if len(H) == n:
                H = np.delete(H, 0)
            l = row["low"]
            L = np.append(L, [l])
            if len(L) == n:
                L = np.delete(L, 0)
            o = row["open"]
            O = np.append(O, [o])
            if len(O) == n:
                O = np.delete(O, 0)
            pc = last_row["close"]
            PC = np.append(PC, [pc])
            if len(PC) == n:
                PC = np.delete(PC, 0)

            ar = (np.sum(np.asarray(H) - np.asarray(O)) / sum(np.asarray(O) - np.asarray(L))) * 100
            AR = np.append(AR, [ar])
            br = (np.sum(np.asarray(H) - np.asarray(PC)) / sum(np.asarray(PC) - np.asarray(L))) * 100
            BR = np.append(BR, [br])

            last_row = row

    return np.asarray(AR), np.asarray(BR)


def dpo(data, n=20, m=6):
    '''
        区间震荡线指标 DPO
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认20
          m:int
              MADPO的参数M，默认6
        return
        -------
          DPO:numpy.ndarray<numpy.float64>
              DPO指标
          MADPO:numpy.ndarray<numpy.float64>
              MADPO指标

    '''

    CLOSES = data["close"]
    DPO = CLOSES - ma(data, int(n / 2 + 1))
    MADPO = _get_any_ma(DPO, m)
    return DPO, MADPO


def trix(data, n=12, m=20):
    import numpy as np

    '''
        三重指数平滑平均线 TRIX
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认12
          m:int
              TRMA的参数M，默认20
        return
        -------
          TRIX:numpy.ndarray<numpy.float64>
              AR指标
          TRMA:numpy.ndarray<numpy.float64>
              BR指标

    '''

    CLOSES = []

    TRIX = []
    for index, row in data.iterrows():
        CLOSES.append(row["close"])

        if len(CLOSES) == n:
            del CLOSES[0]

        tr = np.average(CLOSES)

        if index == 0:
            past_tr = tr
            TRIX.append(0)
        else:

            trix = (tr - past_tr) / past_tr * 100
            TRIX.append(trix)

    TRMA = _get_any_ma(TRIX, m)

    return TRIX, TRMA


def bbi(data):
    import numpy as np

    '''
        Bull And Bearlndex 多空指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
        return
        -------
          BBI:numpy.ndarray<numpy.float64>
              BBI指标

    '''

    CS = []
    BBI = []
    for index, row in data.iterrows():
        CS.append(row["close"])

        if len(CS) < 24:
            BBI.append(row["close"])
        else:
            bbi = np.average([np.average(CS[-3:]), np.average(CS[-6:]), np.average(CS[-12:]), np.average(CS[-24:])])
            BBI.append(bbi)

    return np.asarray(BBI)


def mtm(data, n=6):
    import numpy as np
    '''
        Momentum Index 动量指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认6
        return
        -------
          MTM:numpy.ndarray<numpy.float64>
              MTM动量指标

    '''

    MTM = []
    CN = []
    for index, row in data.iterrows():
        if index < n - 1:
            MTM.append(0.)
        else:
            mtm = row["close"] - CN[index - n]
            MTM.append(mtm)
        CN.append(row["close"])
    return np.asarray(MTM)


def obv(data):
    import numpy as np

    '''
        On Balance Volume 能量潮指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
        return
        -------
          OBV:numpy.ndarray<numpy.float64>
              OBV能量潮指标

    '''

    tmp = np.true_divide(((data["close"] - data["low"]) - (data["high"] - data["close"])), (data["high"] - data["low"]))
    OBV = tmp * data["volume"]
    return OBV


def sar(data, n=4):
    raise Exception("Not implemented yet")


def plot_all(data, is_show=True, output=None):
    import matplotlib.pyplot as plt
    from pylab import rcParams
    import numpy as np
    rcParams['figure.figsize'] = 18, 50

    plt.figure()
    # 收盘价
    plt.subplot(20, 1, 1)
    plt.plot_date(data["date"], data["close"], label="close")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 移动平均线
    plt.subplot(20, 1, 2)
    MA = ma(data, n=10)
    plt.plot_date(data["date"], MA, label="MA(n=10)")
    plt.plot_date(data["date"], data["close"], label="CLOSE PRICE")
    plt.title("MA")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 移动标准差
    n = 10
    plt.subplot(20, 1, 3)
    MD = md(data, n)
    plt.plot_date(data["date"], MD, label="MD(n=10)")
    plt.title("MD")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 指数平均数指标
    plt.subplot(20, 1, 4)
    EMA = ema(data, n)
    plt.plot_date(data["date"], EMA, label="EMA(n=12)")
    plt.title("EMA")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 指数平滑异同平均线(MACD: Moving Average Convergence Divergence)
    plt.subplot(20, 1, 5)
    MACD, DIFF, DEM = macd(data, n)
    plt.plot_date(data["date"], MACD, label="MACD")
    plt.plot_date(data["date"], DIFF, label="DIFF")
    plt.plot_date(data["date"], DEM, label="DEM")
    plt.title("MACD")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 随机指标
    plt.subplot(20, 1, 6)
    K, D, J = kdj(data)
    plt.plot_date(data["date"], K, label="K")
    plt.plot_date(data["date"], D, label="D")
    plt.plot_date(data["date"], J, label="J")
    plt.title("KDJ")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 相对强弱指标
    plt.subplot(20, 1, 7)
    RSI6 = rsi(data, 6)
    RSI12 = rsi(data, 12)
    RSI24 = rsi(data, 24)
    plt.plot_date(data["date"], RSI6, label="RSI(n=6)")
    plt.plot_date(data["date"], RSI12, label="RSI(n=12)")
    plt.plot_date(data["date"], RSI24, label="RSI(n=24)")
    plt.title("RSI")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # BOLL 林线指标
    plt.subplot(20, 1, 8)
    BOLL, UPPER, LOWER = boll(data)
    plt.plot_date(data["date"], BOLL, label="BOLL(n=10)")
    plt.plot_date(data["date"], UPPER, label="UPPER(n=10)")
    plt.plot_date(data["date"], LOWER, label="LOWER(n=10)")
    plt.plot_date(data["date"], data["close"], label="CLOSE PRICE")
    plt.title("BOLL")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # W&R 威廉指标
    plt.subplot(20, 1, 9)
    WNR = wnr(data, n=14)
    plt.plot_date(data["date"], WNR, label="WNR(n=14)")
    plt.title("WNR")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 动向或趋向指标
    plt.subplot(20, 1, 10)
    P_DI, M_DI, ADX, ADXR = dmi(data)
    plt.plot_date(data["date"], P_DI, label="+DI(n=14)")
    plt.plot_date(data["date"], M_DI, label="-DI(n=14)")
    plt.plot_date(data["date"], ADX, label="ADX(m=14)")
    plt.plot_date(data["date"], ADXR, label="ADXR(k=6)")
    plt.title("DMI")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 乖离值
    plt.subplot(20, 1, 11)
    BIAS = bias(data, n=5)
    plt.plot_date(data["date"], BIAS, label="BIAS(n=5)")
    plt.title("BIAS")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 振动升降指标
    plt.subplot(20, 1, 12)
    ASI = asi(data, n=5)
    plt.plot_date(data["date"], ASI, label="ASI(n=5)")
    plt.title("ASI")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 振动升降指标
    plt.subplot(20, 1, 13)
    VR = vr(data, n=26)
    plt.plot_date(data["date"], VR, label="VR(n=26)")
    plt.title("VR")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 振动升降指标
    plt.subplot(20, 1, 14)
    AR, BR = arbr(data, n=26)
    plt.plot_date(data["date"], AR, label="AR(n=26)")
    plt.plot_date(data["date"], BR, label="BR(n=26)")
    plt.title("ARBR")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 区间震荡线
    plt.subplot(20, 1, 15)
    DPO, MADPO = dpo(data, n=20, m=6)
    plt.plot_date(data["date"], DPO, label="DPO(n=20)")
    plt.plot_date(data["date"], MADPO, label="MADPO(m=6)")
    plt.title("DPO")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 三重指数平滑平均线
    plt.subplot(20, 1, 16)
    TRIX, TRMA = trix(data, n=12, m=20)
    plt.plot_date(data["date"], TRIX, label="DPO(n=12)")
    plt.plot_date(data["date"], TRMA, label="MADPO(m=20)")
    plt.title("TRIX")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 多空指标
    plt.subplot(20, 1, 17)
    BBI = bbi(data)
    plt.plot_date(data["date"], BBI, label="BBI(3,6,12,24)")
    plt.title("BBI")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 动量指标
    plt.subplot(20, 1, 18)
    MTM = mtm(data, n=6)
    plt.plot_date(data["date"], MTM, label="MTM(n=6)")
    plt.title("MTM")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    # 动量指标
    plt.subplot(20, 1, 19)
    OBV = obv(data)
    plt.plot_date(data["date"], OBV, label="OBV")
    plt.title("OBV")
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)

    plt.tight_layout()

    if is_show:
        plt.show()

    if output is not None:
        plt.savefig(output)