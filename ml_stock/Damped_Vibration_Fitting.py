#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import numpy as np
import matplotlib
matplotlib.use("Agg")
#matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from scipy import optimize
#阻尼振动 拟合 x = exp(-at)*A*cos(bt + phi)
'''
这里exp是以自然对数为底的指数函数A
a，b，A，phi 由你的阻尼，劲度系数，滑块质量以及初状态决定
'''
def damped_vibration_equation(t,a , b, A, phi,k):
    return A*np.exp(-a*t)*np.cos(b*t+phi)+k*t;
def damped_vibration_equation_fitting(x0, y0,f_linear=damped_vibration_equation ):
    #print '--entering damped_vibration_equation_fitting','长度:',len(x0)==len(y0),len(x0)
    try:
        fitting_result = optimize.curve_fit(f_linear, x0, y0,method='trf',maxfev=5000)
        params = fitting_result[0]
        pcov = fitting_result[1]
        perr = np.sqrt(np.diag(pcov))
        #print pcov,perr,params,len(fitting_result)
        print 'params',params,'std_err',perr
        #if sum(perr)>10:raise
    except Exception,e: 
        #print e,'But we discard them'
        return -1,None
    show_need = False
    if show_need:
        x1=np.array(x0)
        y1 = [f_linear(x_temp,*params) for x_temp in x1]
        #plt.ion() #开启interactive mode不阻塞性显示
        plt.figure()
        #绘制散点
        plt.scatter(x0[:], y0[:], 25, "red")
        plt.plot(x1, y1, "blue")
        plt.title("damped_vibration_equation_fitting")
        plt.xlabel('x')
        plt.ylabel('y')
    next_y = f_linear(x0[-1]+1,*params) 
    _d=(next_y-y0[-1])-0.001#避免超小值的正号提醒
    return _d,plt
if __name__ == '__main__':
    print('模块测试')
    #拟合点
    x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] 
    y0 = [39.986, 38.907, 38.583, 37.461999999999996, 38.533, 38.16, 38.475, 38.55, 38.268, 38.583, 39.338, 40.11, 39.43, 39.513000000000005, 39.77, 39.928000000000004, 39.629, 39.637, 39.611999999999995, 39.355, 39.172, 38.816, 38.143, 37.695, 38.957, 39.196999999999996, 39.239000000000004, 39.637, 40.351, 40.75, 42.842, 43.887, 45.041000000000004, 40.534, 36.483000000000004, 32.839, 29.552, 32.482, 29.576999999999998, 28.355999999999998]
    #x0= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 
    #y0 = [38.957, 39.196999999999996, 39.239000000000004, 39.637, 40.351, 40.75, 42.842, 43.887, 45.041000000000004, 40.534, 36.483000000000004, 32.839, 29.552, 32.482, 29.576999999999998, 28.355999999999998, 27.916, 27.435, 24.695999999999998, 22.23]
    #x0 = [1, 2, 3, 4, 5,8, 18, 36]
    #y0 = [1, 3, 8, 18, 36,8, 18, 36]
    #y0=[damped_vibration_equation(t, a=1, b=1, A=2, phi=0, k=0)for t in x0]
    _d,plt = damped_vibration_equation_fitting(x0,y0)
    #print _d;
    plt.show()


