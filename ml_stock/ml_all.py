#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import data.data_const as dtcst
import ml_const as mlcst
import Naive_Bayes_With_continuous_feature as nv_bys
import hmm_predict as hmm_p
import ml_svm as svm
import numpy as np
import data.util as util
import pandas as pd
import warnings
from data.util import xor
warnings.filterwarnings('ignore')

def ml_all(realtime_data = pd.read_csv(dtcst.My_Store_Dir+'realtime_data.csv').set_index('code')):
    code_func_map = util.code_func_map()
    predict_table = pd.DataFrame(columns=mlcst.attributes)
    basic_env = pd.read_csv(dtcst.My_Store_Dir+'basic.csv',index_col='code')
    predict_table['Code_Index'] = dtcst.My_Wanted_codes
    predict_table = predict_table.set_index("Code_Index")
    i=1;codes = dtcst.My_Wanted_codes;print code_func_map.keys()
    for code in codes:
        print '\n',i,'/',len(codes),'ml_stock',code;i+=1;
        code_rt_dt = realtime_data.loc[int(code)]
        code_basic = basic_env.loc[int(code)]
        try:#code_rt_dt 有时候有重复数据，这里直接加载出来realtime文件没有去重，造成获得series 无法比较 ValueError 
            if code_rt_dt['high'].item()==code_rt_dt['low'].item() and code_rt_dt['open'].item()==0 :#停牌
                continue
            codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
        except:
            continue
        if len(codata)<=25:
            print 'ml_all data amount too small'
            continue
        advice = False;
        try:
            code_funcs = code_func_map[code][0]
            test_method = [func_name_address[name]for name in code_funcs]#0代表是ml_method 控制
        except:
            import ml_stock.not_want_codes_list_object_store as maker
            code_func_map_single = maker.not_want_codes_list_and_code_func_map_generator(issingle_train=True,code=code,not_want_codes_list_needing=True)
            code_func_map[code]=code_func_map_single
            code_funcs = code_func_map[code][0]
            test_method = [func_name_address[name]for name in code_funcs]#0代表是ml_method 控制
            print 'ml_all get code_func_map value error fixed by single train'
        print test_method
        combination_func = lambda codata: [func(codata) for func in test_method]
        aol_func_map =util.aol_func_map
        try:
            code_funcs_logic = code_func_map[code][1]
            real_aols = [aol_func_map[key] for key in code_funcs_logic]
        except: 
            real_aols = [aol_func_map['and'] for key in test_method]
            code_funcs_logic = ['error']
            print 'get real_aol value error'
        ml_combination_result = combination_func(codata)
        #zip函数将以短的长度为准
        for (t,aol) in zip(ml_combination_result,real_aols):
                advice=aol(advice,t)
        info='';
        for (code_func,logic) in zip(code_funcs,code_funcs_logic):
            info=info+ '--'+logic+'--'+code_func
        additional = additional_judge(codata,code_basic,code_rt_dt)
        Coefficient_of_Variation = getCoefficient_of_Variation(codata)
        predict_table.loc[code] = [code_rt_dt['name'],advice,info,additional,Coefficient_of_Variation]
    util.code_func_map(code_func_map)#写回可能更新后的code_func_map
    predict_table.dropna(inplace=True)
    predict_table.to_csv(dtcst.My_Store_Dir+'predict_table.csv')
    recommend_table = recommend_table_generator(predict_table)
    recommend_table.to_csv(dtcst.My_Store_Dir+'recommend_table.csv')
    
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
    if len(codata)<25:return False;
    #可以很好偷懒慎用
    #if codata.tail(1)['ema5_d'].item()<0:return False
    # 训练数据,标签
    #Drop rows by index
    castrate_data = codata.drop([codata.index[len(codata)-1]])
    trainData, labels = nv_bys.getTrainSet(castrate_data)
    features,_ = nv_bys.getTrainSet(codata.tail(1))
    #print '-',type(features),features
    # 该特征应属于哪一类
    result = nv_bys.classify(trainData, labels, features[0])
    #print features,'属于',result  
    return  not '-' in result

def quantity_potiential_judge(codata):#量能判断
    lastdata = codata.tail(1);
    return lastdata['turnover_rate'].item()>mlcst.Turn_Over_rate_thresh and lastdata['amount'].item()>mlcst.Amount_thresh\
            and abs(util.getlast_n_days_p_change(codata=codata,code='None',n=7,end=0))<mlcst.last7days_p_change_thresh and  lastdata['nmc'].item()>mlcst.nmc_thresh
            #流通市值判断#近7日涨跌幅度
#大盘环境用MACD_d值来预估
def env_relative_judge(codata):
    lastdata = codata.tail(1);
    lastdate = lastdata['date']
    env_sh_data = util.get_codata(dtcst.My_Store_Dir,'sh');
    try:
        MACD_d = env_sh_data.loc[lastdate]['MACD_d'].item()
    except:
        print 'env_relative_judg env_sh_data key MACD_d error'
        MACD_d = 0
    return MACD_d>0
#阻尼振动股价拟合
def damped_vibration_fitting(codata):
    #print 'entering damped_vibration_fitting:',
    import Damped_Vibration_Fitting as D_V_F;d_num=40
    if codata.tail(1)['ema5_d'].item()<0:return False
    close = codata.tail(d_num)['close'].tolist()
    c_mean = sum(close)/len(close)
    close=[c- c_mean for c in close]
    X = range(d_num)
    _d,plt = D_V_F.damped_vibration_equation_fitting(X,close)
    try:
        plt.show()
    except: pass
    del plt
    return _d>0
#离散傅里叶分析 预测拟合
def discrete_F_T_predict_fitting(codata):
    #print 'entering discrete_F_T_predict_fitting:',
    import DFT ;d_num=55
    if codata.tail(1)['ema5_d'].item()<0:return False
    close = codata.tail(d_num)['close'].tolist()
    c_mean = sum(close)/len(close)
    close=[c- c_mean for c in close]
    _d,plt = DFT.discrete_F_T_predict(close)
    try:
        plt.show()
    except: pass
    del plt
    return _d>0
def temp_test(codata):
    import Optimized_Curve_fitting as OCF
    if len(codata)%10==0:print len(codata)
    return OCF.get_Optimized_Curver_fitting_func(codata['close'].values)
#便于变相函数持久化,数据量要求小的算法 排在前面
func_name_address = {'default':default_fun,'kdj_ema_quick_d':kdj_ema_quick_d,\
                     'quantity_potiential_judge':quantity_potiential_judge,'macd_diff_judge':macd_diff_judge,\
                    'naive_bayes':naive_bayes,'env_relative_judge':env_relative_judge,\
                    'hmm_state':hmm_state,'ml_svm_predict':ml_svm_predict,'damped_vibration':damped_vibration_fitting,\
                    'discrete_F_T_predict_fitting':discrete_F_T_predict_fitting,'temp_test':temp_test}
def get_optimized_func_of_codes_by_sequence_similarity(codata):
    import copy
    #print 'entering get_optimized_func_of_codes_by_sequence_similarity:',
    d_num=25;y=codata.tail(d_num)['close'].tolist();   wanted_index=0;max_cos=-1;
    previous_codata = codata.drop([codata.index[-1]])
    for i in range(len(previous_codata)-d_num):
        x = previous_codata.head(i+d_num).tail(d_num)['close'].tolist();
        temp_cos=util.vector_cos(x, y)
        if temp_cos>max_cos:
            max_cos=temp_cos;wanted_index=i
    temp_codata = codata
    temp_func_map = copy.copy(func_name_address)
    for i in range(wanted_index,len(codata)):
        if len(temp_func_map)==1: break
        else: 
            list_keys = list(temp_func_map.keys())
            for key in list_keys:
                func=temp_func_map[key] #RuntimeError: dictionary changed size during iteration
                if len(temp_func_map)<=1: 
                    temp_func_map ={ temp_func_map.keys()[0]:temp_func_map[temp_func_map.keys()[0]]}
                    break
                else:
                    if i+d_num+1>=len(temp_codata):
                        temp_func_map = {temp_func_map.keys()[0]:temp_func_map[temp_func_map.keys()[0]]}
                        break
                    predictor_codata =temp_codata.head(i+d_num); y_codata = temp_codata.head(i+d_num+1).tail(1)
                    #print func(predictor_codata),y_codata['p_change'].item()
                    #print predictor_codata.tail(1),y_codata
                    if xor(func(predictor_codata), y_codata['p_change'].item()>0):
                        #print 'temp_func_map.pop(key)'
                        temp_func_map.pop(key)
        #print len(temp_func_map),temp_func_map.keys()
    wanted_func_name = temp_func_map.keys()[0]
    print 'leaving:',wanted_func_name
    wanted_func = temp_func_map[wanted_func_name]
    return wanted_func(codata)
#自己选股的附加限制避免推荐过多无法选择，优秀性未经检验
def additional_judge(codata,code_basic,code_rt_dt):
    return isnortheast(code_basic) and isst_stock(code_rt_dt) and in_market_days_mul_macd_d_greater_0_num(codata)
#东北企业筛选限制
def isnortheast(code_basic):
    return not code_basic['area']== '黑龙江' and not code_basic['area']== '吉林' and not code_basic['area']== '辽宁'
def isst_stock(code_rt_dt):
    return not 'ST' in code_rt_dt['name']
#macd天数和上市天数的积 选择偏向次新和macd少的 附加一个成交额门限
def in_market_days_mul_macd_d_greater_0_num(codata):
    lastdata = codata.tail(1);last_judge = lastdata['MACD_d'].item()>0 and lastdata['p_change'].item()>0
    num = 0;in_market_days_mul_macd_d_greater_0_num_thresh = 2**12
    for index,row in codata.tail(5).iterrows():
        if row['MACD_d']>0:num+=1
    return last_judge and num*len(codata)<in_market_days_mul_macd_d_greater_0_num_thresh and lastdata['amount'].item()>mlcst.Amount_thresh\
        and abs(lastdata['p_change'].item())<6

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

def recommend_table_generator(predict_table):
    recommend_table = predict_table.drop(predict_table.index)#生成一个只有列名的空dataframe
    print 'entering recommend_table_generator predict_table len: ',len(predict_table)
    for Code_Index,item in predict_table.iterrows():
        if int(Code_Index)<599999:
            continue
        advice = item['advice']
        if advice:
            recommend_table.loc[Code_Index] = item
    df = recommend_table
    print 'after advice judge recommend_table len: ',len(recommend_table)
    mean__Coefficient_of_Variation = predict_table['Coefficient_of_Variation'].mean()
    #单独构造 series 原因是： Naive_Bayes 的比较不能使用相应比较的==或者!=符号
    #Naive_Bayes_select = pd.Series([not item.find('-')>-1for item in df['Naive_Bayes']], index=df.index.tolist())
    #shorterm_fitting_select = pd.Series([abs(item)<mlcst.shorterm_fitting_thresh for item in df['Shorterm_Fitting']], index=df.index.tolist())
    #A common operation is the use of boolean vectors to filter the data. 
    #The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses--().
    recommend_table = df[( (df['additional_judge']==True) & (df['Coefficient_of_Variation']>=mean__Coefficient_of_Variation) )]
    special_list = ['603590','603259','600036']
    for Code_Index in special_list:
        try:
            predict_table.loc[Code_Index]['Coefficient_of_Variation'] = 1;
            recommend_table.loc[Code_Index] = predict_table.loc[Code_Index]
        except:pass
    recommend_table.sort_values(by="Coefficient_of_Variation" , ascending=False,inplace=True)
    print 'after additional judge recommend_table len: ',len(recommend_table)
    return util.type_data(recommend_table.head(10+len(special_list)))

if __name__ == '__main__':
    print '模块测试'
    ml_all()