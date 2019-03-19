#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.dates as mdt
import matplotlib.pyplot as plt
#import matplotlib.finance as mpf
import tushare as ts
import seaborn as sns
import data.data_const as dtcst
#参考 ：http://www.cnblogs.com/timotong/p/9501430.html数据可视化
def line_chart():#折线图
    plt.figure(figsize=(16,8)) #定义图的大小
    plt.plot(df[['turnover_rate','close']].tail(150)) #定义数据，可以是一个数据，也可以是两个数据，可以是dataframe或者series格式，也可以是list或者numpy.array格式
    plt.xlabel("xxx") #所有种类的图都可以添加x、y轴标签和名称，没有也没关系plt.ylabel("xxx")plt.title("xxx")
    plt.show() #显示图

def plt_subplot():#同x轴，
    plt.figure(figsize=(75,55))
    plt.subplot(211) #括号里的（mnp）：m表示是图排成m行，n表示图排成n列，p表示位置
    plt.plot(df['MACD_d_d'])
    plt.subplot(212)
    plt.plot(df['close'])
    plt.show()

def scatter_point(df):#散点图'关联可视化,用matpltlib.scatter方法对两组数据进行plot,看是否由明显线性关系.'
    plt.figure(figsize=(75,55))
    plt.scatter(df.index,df['volume'],c='r')
    plt.show()

def histogram(df):#直方图
    plt.figure(figsize=(75,55))
    plt.hist(df['MACD_d_d'])
    plt.show() 
    
def candle_pic():
    print  '调整次序到ohlc'
    print  '调整DateTime格式到float(网上有类似的转换方法,可能基于不同的版本,在我的环境下没有成功)'
    histd['date']= pd.to_datetime(histd['date']) 
    print  ' convert str to date'
    histd['date'] = mdt.date2num(histd['date'].astype(dt.date))  
    print  'convert date to float days'
    #print histd
    print  '转换DataFrame为Numpy.Array(网上有重新构建tuple的方法,我觉得转换成Numpy Array更直接)'
    hista = np.array(histd)
    #print hista
    print  '利用hista画出蜡烛图'
    fig,ax=plt.subplots(figsize=(75,55))
    fig.subplots_adjust(bottom=0.5)
    #mpf.candlestick_ochl(ax,hista,width=0.6,colorup='r',colordown='g',alpha=1.0) # ax 绘图Axes的实例，data价格历史数据 width 图像中红绿矩形的宽度,代表天数,colorup 收盘价格大于开盘价格时的颜色,colordown 低于开盘价格时矩形的颜色,alpha 矩形的颜色的透明度
    plt.grid(True) #是否显示格子
    plt.title('k_chart')
    plt.xlabel('date')
    plt.ylabel('price')
    ax.xaxis_date() # x轴的刻度为日期
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(),rotation=30)
    plt.show()
def _1_1_func(num):
    if num>0:return 1
    elif num<0:return -1
    else:return 0
code='603955'
df = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix)
df[['close','ema_quick']].plot()
df=df.tail(250)
index = 'ema_quick_d'
histd = df[['review_pc',index]]

'''
#预处理数据
df = histd 
for idx in df.index.tolist():
        df.loc[idx,index] = _1_1_func(df.loc[idx,index])
histd = df'''
'''去除可能异常值的预处理
a=np.array( abs(df[index]) )
MACD_d_thresh = np.percentile(a,99)*1.5#99%分位数
MACD_d_select = pd.Series([abs(item)<MACD_d_thresh for item in df[index]], index=df.index.tolist())
print [item==False for item in MACD_d_select],len(MACD_d_select)
histd = df[( (MACD_d_select) & (MACD_d_select) )]
'''

'''
print  '表示位置的统计量:平均值和中位数'
print histd.median()#中位数
print  '表示数据散度的统计量:标准差,方差和极差'
print  'DataFrame不支持ptp()'
print np.ptp(histd['open']) #极差———最大值与最小值之差:ptp()
print histd.var()
print  '表示分布形状的统计量: 偏度和峰度'
print histd.skew()
print histd.kurt()
print  '分布可视化,用hist方法,用法和plot一样'
print  '要注意不能有nan值,否则报错'
histd['p_change'].plot(figsize=(75,55), grid=True) 
plt.show()
print  '箱体图'
histd[['p_change','turnover_rate']].boxplot()
plt.show()''' 
print  '散点矩阵'
print  '对于多维数据可以两两配对组成散点图矩阵非常直观的展示数据之间的相关性.现成方法是引入seaborn库'
sns.set()
print  'Pandas的DataFrame格式的数据'
print  'species是用来分类的列'
sns.pairplot(histd, hue='review_pc')
print 'plt后端',plt.get_backend()
plt.show()