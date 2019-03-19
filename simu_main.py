#!/usr/bin/env python
# -*- coding: utf-8 -*-
#引入data,ml_stock 包，解决包导入问题
import data,ml_stock
import ml_stock.simulation as simu
import ml_stock.not_want_codes_list_object_store as dislike
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
        dislike.not_want_codes_list_and_code_func_map_generator()
'''
Created on 2018年8月5日
@author: feiyu
'''
