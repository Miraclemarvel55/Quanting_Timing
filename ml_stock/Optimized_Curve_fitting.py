# -*- coding:utf-8 -*-

import numpy as np
import data.util as util
#阻尼振动 拟合 x = exp(-at)*A*cos(bt + phi) 这里exp是以自然对数为底的指数函数A ，a，b，A，phi 由你的阻尼，劲度系数，滑块质量以及初状态决定
def damped_vibration_equation(t,a , b, A, phi):
    return A*np.exp(-a*t)*np.cos(b*t+phi);
#a*a/2*np.cos(t*omg)+b*b*np.exp(-a*abs(t)) 不需要绝对值，信号与噪声 随机过程 
def signal_and_noise_equation(t,a,omg,b):
    return a*a/2*np.cos(t*omg)+b*b*np.exp(-a*t)
def cos_or_sin(t,a,phi,w,incept):
    return a*np.cos(t*w+phi)+incept

#非稳定过程函数
def linear_plus_cos(t,k,incept,a,w,phi):
    return k*t+incept+a*np.cos(w*t+phi)
func_mapping ={'damped_vibration_equation':damped_vibration_equation,'signal_and_noise_equation':signal_and_noise_equation,\
               'linear_plus_cos':linear_plus_cos}
def func_fitting(x0, y0,f_linear=damped_vibration_equation,title='curve_fitting'):
    try:
        from scipy import optimize;y_fit=None
        fitting_result = optimize.curve_fit(f_linear, x0, y0,method='trf',maxfev=5000)
        params = fitting_result[0];y_fit = [f_linear(x_temp,*params) for x_temp in x0]
        '''pcov = fitting_result[1]
        perr = np.sqrt(np.diag(pcov))'''
    except Exception,e: 
        #print(e,'exception in func_fitting',y0) #超过设定的拟合最大次数
        return -1,None,0
    show_need = False;plt=None
    if show_need:
        import matplotlib
        matplotlib.use("Agg")
        #matplotlib.use("Pdf")
        import matplotlib.pyplot as plt
        plt.close()
        x1=np.array(range(min(x0)*10,(max(x0)+3)*10,1))/10.0
        y1 = [f_linear(x_temp,*params) for x_temp in x1]
        #plt.ion() #开启interactive mode不阻塞性显示
        plt.figure()
        #绘制散点
        plt.scatter(x0[:], y0[:], 25, "red")
        plt.plot(x1, y1, "blue")
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
    next_y = f_linear(x0[-1]+0.25,*params) 
    _d=(next_y-y_fit[-1])-0.01#避免超小值的正号提醒
    r2 = util.get_r2_numpy(y_fit=y_fit, y=y0); fotcc = util.get_first_order_temporal_correlation_coefficient(y_fit, y0)
    goodness_of_fitting = (r2+fotcc)/2;
    goodness_of_fitting = (r2)
    return _d,plt,goodness_of_fitting

def get_Optimized_Curver_fitting_func(Y):
    max_limit = 35
    min_i=30;max_i=len(Y)if len(Y)<max_limit else max_limit;goodness_of_fitting_thresh = 0.85
    close = Y
    for i in range(min_i,max_i):
        #print i,
        y0 = util.midlelize(close[-i:]);y0 = y0/np.std(y0);
        x0=range(i);goodness0_1= goodness_of_fitting_thresh;wanted_d=0;wanted_key=None;
        for key,func in func_mapping.iteritems():
            _d,plt,goodness_of_fitting = func_fitting(x0,y0,func,key);
            if goodness0_1<goodness_of_fitting:
                goodness0_1=goodness_of_fitting;wanted_d=_d;wanted_key = key
                try:print key,'goodness0_1',goodness0_1,'_d',_d;plt.show();
                except:pass
            else:plt=None
        if goodness0_1>goodness_of_fitting_thresh:
            print '---',goodness0_1,wanted_key,'wanted_d',wanted_d
            return wanted_d>0
    return False
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
    #signal = get_Optimized_Curver_fitting_func(y0)
    _d,plt,perr = func_fitting(x0, y0, f_linear=cos_or_sin);print _d;plt.show()
        
        
        
        