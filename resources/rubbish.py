#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys,os

# 单独构造 series 原因是： Naive_Bayes 的比较不能使用相应比较的==或者!=符号
# Naive_Bayes_select = pd.Series([not item.find('-')>-1for item in df['Naive_Bayes']], index=df.index.tolist())
# shorterm_fitting_select = pd.Series([abs(item)<mlcst.shorterm_fitting_thresh for item in df['Shorterm_Fitting']], index=df.index.tolist())
# A common operation is the use of boolean vectors to filter the data.
# The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses--().

'''plt.subplot(211)#括号里的（mnp）：m表示是图排成m行，n表示图排成n列，p表示位置
plt.plot(codata[['pseudo_mean_price_PVT']].tail(100));plt.grid()  # 生成网格
plt.subplot(212)
plt.plot(codata['close'].tail(100));plt.grid()  # 生成网格
plt.savefig(dtcst.cweb_path+'analyzer_result_pic/'+code+'.png');
plt.cla() # 清坐标轴。
plt.clf() # 清图。
plt.close() # 关窗口
gc.collect();'''

# import cv2 as cv
# import numpy as np
# #-----------二值化（黑0和白 255）-------------
# #二值化的方法（全局阈值  局部阈值（自适应阈值））
# # OTSU
# #cv.THRESH_BINARY 二值化
# #cv.THRESH_BINARY_INV(黑白调换)
# #cv.THRES_TRUNC 截断
#
# def threshold(img):  #全局阈值
#     gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)  #首先变为灰度图
#     ret , binary = cv.threshold( gray , 0, 255 , cv.THRESH_BINARY |cv.THRESH_OTSU)#cv.THRESH_BINARY |cv.THRESH_OTSU 根据THRESH_OTSU阈值进行二值化  cv.THRESH_BINARY_INV(黑白调换)
#     #上面的0 为阈值 ，当cv.THRESH_OTSU 不设置则 0 生效
#     #ret 阈值 ， binary二值化图像
#     print("阈值：", ret)
#     cv.imshow("binary", binary)
#
# def own_threshold(img): #自己设置阈值100            全局
#     gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)  #首先变为灰度图
#     ret , binary = cv.threshold( gray , 250, 255 , cv.THRESH_BINARY )#cv.THRESH_BINARY |cv.THRESH_OTSU 根据THRESH_OTSU阈值进行二值化
#     #上面的0 为阈值 ，当cv.THRESH_OTSU 不设置则 0 生效
#     #ret 阈值 ， binary二值化图像
#     print("阈值：", ret)
#     cv.imshow("binary", binary)
#     b = binary.astype(str)
#     b[b=="255"] = "*"
#     # b[b=="0"]  = "0"
#     print(b.shape)
#     np.savetxt('a.csv',b,fmt='%c',delimiter=',')
#
# def local_threshold(img):  #局部阈值
#
#     gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)  #首先变为灰度图
#     binary = cv.adaptiveThreshold( gray ,255 , cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, 25 , 10,)#255 最大值
#     #上面的 有两种方法ADAPTIVE_THRESH_GAUSSIAN_C （带权重的均值）和ADAPTIVE_THRESH_MEAN_C（和均值比较）
#     #blockSize 必须为奇数 ，c为常量（每个像素块均值 和均值比较 大的多余c。。。少于c）
#     #ret 阈值 ， binary二值化图像
#     cv.imshow("binary", binary)
#
# def custom_threshold(img):  #自己计算均值二值化
#     gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)  #首先变为灰度图
#     h ,w = gray.shape[:2]
#     m = np.reshape( gray ,[1 ,w+h])
#     mean = m.sum() / w*h  #求出均值
#     binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY )
#     cv.imshow("binary", binary)
#
#
# def main():
#     img = cv.imread("1.jpg")
#     x,y = img.shape[0:2]
#
#     factor = 8
#     img = cv.resize(img,(x//factor,y//factor),interpolation=3L)
#     # cv.namedWindow("Show", cv.WINDOW_AUTOSIZE)
#     cv.imshow("Show", img)
#
#     own_threshold(img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     print("exited over")
#
# main()