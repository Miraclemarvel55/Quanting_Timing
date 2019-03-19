#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#外部调用函数
def get_last_state(codata):
	state = True
	X ,multy_array= GetDataForHMM(codata);close = multy_array[1]
	
	HMM_model = getHMM_fit_model(X)
	hidden_states = HMMPredict(HMM_model, codata)
	mst_av_hd_stt = get_most_avail_hidden_state(hidden_states, close, HMM_model)
	tail_num = 10#一定要大于5呀，原因请看 GetDataForHMM(codata)实现
	predictResult = HMMPredict(HMM_model, codata.tail(tail_num))[-1]
	if predictResult not in mst_av_hd_stt:
		state = False
	return state,getplt_show(hidden_states, HMM_model, multy_array)

def GetDataForHMM(codata):
	#print '--entering GetDataForHMM:',type(codata)#codata.columns
	turnover_rate = codata['turnover_rate']
	turnover_rate = pd.DataFrame(turnover_rate)
	tradeDate = pd.to_datetime(codata['date'][5:])#日期列表
	volume = codata['volume'][5:]#2 成交量数据
	volume = np.array(volume)
	close = codata['close'] # 3 收盘价数据
	delta_h_l = np.log(np.array(codata['high'])) - np.log(np.array(codata['low'])) #3 当日对数高低价差
	delta_h_l = delta_h_l[5:]#振幅
	#取对数进行计算，np.diff()后一个数据减去前一个数据的差
	logReturn1 = np.array(np.diff(np.log(close))) #4 对数股价差
	logReturn1 = logReturn1[4:]
	logReturn5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))# 5日 对数收益差
	log_turnover_rate = np.array(np.diff(np.log(turnover_rate[turnover_rate.columns[0]])))[4:]
	close = close[5:]
	X = np.column_stack([logReturn1,logReturn5,delta_h_l,volume,log_turnover_rate]) # 将几个array合成一个2Darray
	multy_array = (tradeDate,close)
	#print '--leaving GetDataForHMM:',len(X)
	return X ,multy_array#注意第一列的，tradeDate数据与其他列的数据格式不一样
#利用上证指数更新马尔科夫模型
def getHMM_fit_model(X):
	states_num = 5
	param=set(X.ravel())#参考自stack overflow网友
	HMM_model = GaussianHMM(n_components=states_num, covariance_type="diag", n_iter=800,params=param).fit(X)
	return HMM_model
def HMMPredict(HMM_model,codata):
	#print '--entering HMMPredict:',type(HMM_model),type(codata),len(codata)
	X,_= GetDataForHMM(codata)
	hidden_states = HMM_model.predict(X)
	return hidden_states
#图形显示
def getplt_show(hidden_states,HMM_model,multy_array,show_needing=False):
	if not show_needing:
		return
	from matplotlib import pyplot as plt
	show_nums = 100
	tradeDate = multy_array[0].tail(show_nums);close = multy_array[1].tail(show_nums);hidden_states=hidden_states[len(hidden_states)-1-show_nums:hidden_states.size-1]
	#tradeDate = multy_array[0];close = multy_array[1]
	plt.figure(figsize=(15, 8)) 
	for i in range(HMM_model.n_components):#model.n_components=6代表6个不同的隐藏状态用i遍历6个状态
		idx = (hidden_states==i) #如果hidden_states==i则返回true 否者返回false
		#print type(idx),idx.size,tradeDate[idx].size,close[idx].size
		plt.plot_date(tradeDate[idx],close[idx],label='%dth hidden state'%i,lw=1) 
		#plt.plot_date画时间，当hidden_states==i 那天为ture时讲收盘价画上去组成一条线总共有6跳线进行叠加第一个参数时间，
		#后面的Y轴数值，最新日期，收盘指数  画的状态线是'.'点状 %d 第%i种状态，lw=1 线粗=1
	plt.legend()    #显示图例，就是右上角的标签
	plt.grid()     #显示网格
	return plt
def get_most_avail_hidden_state(hidden_states,close,HMM_model):
	perform_list = []
	for i in range(HMM_model.n_components):#model.n_components=6代表6个不同的隐藏状态用i遍历6个状态
		asset = 0;ishold = 0; sellprice = 0
		idx = (hidden_states==i) #如果hidden_states==i则返回true 否者返回false
		index = close[idx].index
		for idx in range(index.size):
			if ishold == 0:
				asset -=close.loc[index[idx]]
				sellprice = close.loc[index[idx]]
				ishold=1
			else:
				sellprice=close.loc[index[idx]]
				if index[idx]!=index[idx-1]+1:
					asset+=sellprice
					ishold=0
				
		if len(index)!=0:
			if ishold==1:
				asset +=sellprice
		else:asset=-999
		perform_list +=[asset]
	mst_av_hd_stt_0 = perform_list.index(max(perform_list))	
	perform_list[mst_av_hd_stt_0]=min(perform_list)
	mst_av_hd_stt_1 = perform_list.index(max(perform_list))	
	return (mst_av_hd_stt_0,mst_av_hd_stt_1)	
if __name__ == '__main__':
	import data.data_const as data_const
	for code in data_const.Test_Wanted_codes:
		#code = '000002'
		codata = pd.read_csv(data_const.My_Database_Dir+code+data_const.Suffix)
		X ,multy_array= GetDataForHMM(codata);close = multy_array[1]
		HMM_model = getHMM_fit_model(X)
		hidden_states = HMMPredict(HMM_model, codata)
		print get_most_avail_hidden_state(hidden_states, close, HMM_model)
		
		predictResult = HMMPredict(HMM_model, codata.tail(10))
		print predictResult[-1]
		getplt_show(hidden_states, HMM_model, multy_array).show()
