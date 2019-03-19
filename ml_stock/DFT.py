#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 函数功能：将频域数据转换成时序数据
# bins为频域数据，n设置使用前多少个频域数据，loop设置生成数据的长度
def fft_extend(bins, n, loops=1):
    length = int(len(bins) * loops)
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(bins[:n]):
        if k != 0 : p *= 2 # 除去直流成分之外, 其余的系数都 * 2
        data += np.real(p) * np.cos(k*index) # 余弦成分的系数为实数部分
        data -= np.imag(p) * np.sin(k*index) # 正弦成分的系数为负的虚数部分
    return index, data
def discrete_F_T_predict(ts_log):
    ts_diff = np.diff(ts_log,1)
    #ts_diff = np.dropna(ts_diff)
    fy = np.fft.fft(ts_diff); len_fy=len(fy)
    index, conv2 = fft_extend(fy / len(ts_diff), int(len(fy)/2-1), loops=1.1) # 0到N/2 频率
    show_need = False
    if show_need:
        conv1 = np.real(np.fft.ifft(fy)) # 逆变换
        print(conv1,len(conv1) )
        print index,conv2,len(index),len(conv2),conv2[len_fy]
        plt.plot(ts_diff)
        plt.plot(conv1 - 0.5) # 为看清楚，将显示区域下拉0.5import matplotlib.pyplot as plt
        plt.plot(conv2 - 1)
    _d = conv2[len_fy]-ts_log[-1]-0.001
    return _d,plt
if __name__ == '__main__':
    ts = np.cos(range(0,21,))
    #ts_log = np.log(ts)# 平稳化 不平稳序列需要
    ts_log = ts
    ts_diff = np.diff(ts_log,1)
    #ts_diff = np.dropna(ts_diff)
    fy = np.fft.fft(ts_diff)
    conv1 = np.real(np.fft.ifft(fy)) # 逆变换
    print(conv1,len(conv1) )
    index, conv2 = fft_extend(fy / len(ts_diff), int(len(fy)/2-1), loops=1.1) # 只关心一半数据
    print index,conv2,len(index),len(conv2),conv2[len(fy)]
    plt.plot(ts_diff)
    plt.plot(conv1 - 0.5) # 为看清楚，将显示区域下拉0.5
    plt.plot(conv2 - 1)
    plt.show()
    
    