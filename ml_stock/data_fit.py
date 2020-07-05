#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import numpy as np
import matplotlib
matplotlib.use("Agg")
#matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from scipy import optimize

#直线方程函数
def f_linear(x, A, B):
    return A*x + B
def linear_fitting(f_linear, x0, y0):
    #print '--entering linear_fitting',x0,y0,'长度:',len(x0)==len(y0),len(x0)
    #直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_linear, x0, y0)[0]
    #x1 = np.arange(0, 6, 0.01)
    x1=np.array(x0)
    y1 = A1*x1 + B1
    show_need = False
    if show_need:
        #plt.ion() #开启interactive mode不阻塞性显示
        plt.figure()
        #绘制散点
        plt.scatter(x0[:], y0[:], 25, "red")
        plt.plot(x1, y1, "blue")
        plt.title("linear_fitting")
        plt.xlabel('x')
        plt.ylabel('y')
    next_y = A1*(x0[-1]+1)+B1
    _d=(next_y-y0[-1])
    return _d,plt
#二次曲线方程
def f_quadratic(x, A, B, C):
    return A*x*x + B*x + C

#三次曲线方程
def f_triple(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D

#正态拟合
def f_gauss(x, A, B, C, sigma):
    return A*np.exp(-(x-B)**2/(2*sigma**2)) + C
#阻尼振动 拟合 x = exp(-at)*A*cos(bt + phi)
#这里exp是以自然对数为底的指数函数
#a，b，A，phi 由你的阻尼，劲度系数，滑块质量以及初状态决定
def damped_vibration_equation(t,a , b, A, phi):
    import numpy as np;import math
    return A*np.exp(-a*t)*math.cos(b*t+phi);

def gauss_fitting(f_gauss, x0, y0,wanted_x):
    print '-entering gauss_fitting',x0,y0,'长度:',len(x0)==len(y0),len(x0)
    #直线拟合与绘制
    A, B,C,sigma = optimize.curve_fit(f_gauss, x0, y0)[0]
    #x1 = np.arange(0, 6, 0.01)
    x1=np.array(x0)
    y1 = A*np.exp(-(x1-B)**2/(2*sigma**2)) + C
    #plt.ion() #开启interactive mode不阻塞性显示
    plt.figure()
    #绘制散点
    plt.scatter(x0[:], y0[:], 25, "red")
    plt.plot(x1, y1, "blue")
    plt.title("gauss_fitting")
    plt.xlabel('x')
    plt.ylabel('y')
    wanted_x_density = A*np.exp(-(wanted_x-B)**2/(2*sigma**2)) + C
    return wanted_x_density,plt

def plot_test():

    plt.figure()

    #拟合点
    x0 = [1, 2, 3, 4, 5]
    y0 = [1, 3, 8, 18, 36]

    #绘制散点
    plt.scatter(x0[:], y0[:], 25, "red")

    #直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_linear, x0, y0)[0]
    x1 = np.arange(0, 6, 0.01)
    y1 = A1*x1 + B1
    plt.plot(x1, y1, "blue")

    #二次曲线拟合与绘制
    A2, B2, C2 = optimize.curve_fit(f_quadratic, x0, y0)[0]
    x2 = np.arange(0, 6, 0.01)
    y2 = A2*x2*x2 + B2*x2 + C2 
    plt.plot(x2, y2, "green")

    #三次曲线拟合与绘制
    A3, B3, C3, D3= optimize.curve_fit(f_triple, x0, y0)[0]
    x3 = np.arange(0, 6, 0.01)
    y3 = A3*x3*x3*x3 + B3*x3*x3 + C3*x3 + D3 
    plt.plot(x3, y3, "purple")

    plt.title("equation fit test")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()    
def damped_vibration_equation_scatter(a , b, A, phi):
    t = np.arange(0, 20, 0.1)
    x = [damped_vibration_equation(temp,a , b, A, phi) for temp in t]
    plt.scatter(t, x, 25, "red")
    plt.show()
if __name__ == '__main__':
    print('模块测试')
    #plot_test()
    #damped_vibration_equation_scatter(a=1,b= 5.5, A=2,phi= 0)
    damped_vibration_equation_scatter(*[-0.06425622 , 1.84890683, -8.06971788, 0.52492982] )
    
    
    