#!/usr/bin/env python
# -*- coding: utf-8 -*-
#引入data,ml_stock 包，解决包导入问题
import data
import ml_stock.ml_all as ml_all
import ml_stock.simulation as simu
import warnings
warnings.filterwarnings('ignore')

#用于命令行下测试方法，可以获取全部打印信息，及服务器测试
if __name__ == '__main__':
    #simu.ml_funcs_perform_compare()
    ml_all.ml_all()
'''
Created on 2018年8月5日

@author: feiyu
'''
