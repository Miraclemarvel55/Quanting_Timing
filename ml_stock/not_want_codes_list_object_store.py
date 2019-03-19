#!/usr/bin/env python
# -*- coding: utf-8 -*-
import data.data_const as dtcst
from ml_stock import simulation
from data import util
import warnings
warnings.filterwarnings('ignore')

review_num_train=20
def not_want_codes_list_and_code_func_map_generator(issingle_train=False,code=None,not_want_codes_list_needing=False,review_num=review_num_train,review_end=0):
    if issingle_train ==True:
        code_func_map,_ = simulation.get_optimized_func_of_codes(review_num, review_end, [code])
        try:
            return code_func_map[code]
        except:
            import ml_all
            code_func_map[code] = [ml_all.func_name_address.keys()[:3],util.aol_func_map.keys()[:3]]#随便给的,数据量要求小的ml算法
            return code_func_map[code]
      
    codes=dtcst.Whole_codes
    print 'codes_nums',len(codes)
    if not_want_codes_list_needing:
        _,data1,_,_=simulation.whole_stock_ml_perform_simulation_of_func_code(review_num=review_num_train,review_end=0,codes=codes) 
        myflatten = lambda x: [y for l in x for y in myflatten(l)] if type(x) is list else [x]
        data1=myflatten(data1);print 'not_like_codes_list:',data1
        output_file = open(dtcst.Project_Root_Dir+"not_like_codes_list",'wb')#把这些数据序列化到文件中，注：文件必须以二进制模式打开
        import marshal
        marshal.dump(data1,output_file)
        output_file.close()   
             
    code_func_map,_ = simulation.get_optimized_func_of_codes(review_num_train, 0, codes)       
    util.code_func_map(code_func_map);print len(code_func_map)  
if __name__ == '__main__':
    not_want_codes_list_and_code_func_map_generator()