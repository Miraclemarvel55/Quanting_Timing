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
'''#短期股价曲线拟合限制-----废弃
def shorterm_fitting(codata):
    #print 'entering shorterm_fitting:',
    d_num=3
    close = codata.tail(d_num)['close'].tolist()
    
    X = range(d_num)
    _d,_ = data_fit.linear_fitting(data_fit.f_linear,X,close)
    #print 'shorterm_fitting',_d
    #plt.show()
    if _d>mlcst.shorterm_fitting_thresh:
        return False
    else : return True
'''
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