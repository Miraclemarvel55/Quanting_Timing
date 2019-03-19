#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#code_self_ml_attributes = ['Code_Index', 'Name','MACD_DIFF_Judge', 'HMM_State', 'Naive_Bayes','self_experiences','additional_judge']
code_self_ml_attributes = ['Code_Index', 'Name','advice','info']+['additional_judge']
#Coefficient of Variation 变异系数，标准差除以均值，也可以用方差除以均值平方近似，相差平方关系
#last7days_p_change 最近7日涨幅 in_market_days_num 上市天数
as_env_attributes = ['Coefficient_of_Variation']
#code_self_realtime_attributes = ['turnover_rate','amount']
#attributes = code_self_ml_attributes+as_env_attributes+code_self_realtime_attributes
attributes = code_self_ml_attributes + as_env_attributes

#咸鱼理论状态名
dead = "死鱼";
head = "鱼头";
jump = "鱼跃";
body = "鱼身";
tail = "鱼尾";
#nmc 单位万元
nmc_thresh=25*10**4;Turn_Over_rate_thresh =0.5;Amount_thresh = 150*100*10000;shorterm_fitting_thresh = 0;
last7days_p_change_thresh = 15;

