#!/usr/bin/env python
# -*- coding: utf-8 -*-
import marshal
import data.data_const as dtcst
import simulation2
def not_want_codes_list_and_code_func_map_generator():
    codes=dtcst.Whole_codes
    print 'codes_nums',len(codes)
    _,data1,_,_=simulation2.whole_stock_ml_perform_simulation_of_func_code(review_num=10,review_end=0,codes=codes) 
    myflatten = lambda x: [y for l in x for y in myflatten(l)] if type(x) is list else [x]
    data1=myflatten(data1);print 'not_like_codes_list:',data1
    output_file = open(dtcst.Project_Root_Dir+"not_like_codes_list",'wb')#把这些数据序列化到文件中，注：文件必须以二进制模式打开
    marshal.dump(data1,output_file)
    output_file.close()        
    
    code_func_map,_ = simulation2.get_optimized_func_of_codes(15, 0, codes)       
    try:
        output_file = open(dtcst.Project_Root_Dir+"code_func_map",'wb')#把这些数据序列化到文件中，注：文件必须以二进制模式打开
        marshal.dump(code_func_map,output_file)
        output_file.close()
        print '\ncode_func_map write over'
    except:
        print 'code_func_map 持久化 失败'       
    
    '''    review_num=simulation.get_optimized_period_review_num_of_simulation(codes=codes)

    input_file = open('mydbase','rb')#从文件中读取序列化的数据
    #data1 = []
    data1 = marshal.load(input_file)
    data2 = marshal.load(input_file)
    data3 = marshal.load(input_file)
    print data1#给同志们打印出结果看看
    print data2
    print data3
     '''
    ''' 
    outstring = marshal.dumps(data1)    #marshal.dumps()返回是一个字节串，该字节串用于写入文件
    open('mydbase','wb').write(outstring)
     
     
    file_data = open('mydbase','rb').read()
    real_data = marshal.loads(file_data)
    print real_data'''
    '''
        import shelve  
        dbase = shelve.open("mydbase")  
        object1 = ['The', 'bright', ('side', 'of'), ['life']]  
        object2 = {'name': 'Brian', 'age': 33, 'motto': object1}  
        dbase['brian']  = object2  
        dbase['knight'] = {'name': 'Knight', 'motto': 'Ni!'}  
        dbase.close( )
        dbase = shelve.open("mydbase")
        print len(dbase),dbase.keys( ),dbase['brian']
    '''
if __name__ == '__main__':
    not_want_codes_list_and_code_func_map_generator()