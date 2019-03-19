def turnoverratio_filter(context,security_list,daft,threshold1,threshold2):
    new_stock_list = []
    trade_days = jqdata.get_trade_days(end_date=context.previous_date, count = daft)
    for security in security_list:
        turnover_sum=0
        countdays=daft
        for trade in trade_days:
            q = query(valuation.turnover_ratio).filter(valuation.code == security)
            turnover = get_fundamentals(q, trade)['turnover_ratio']
            if np.array(turnover) > 0:
                pass
            else:
                countdays = countdays - 1
            turnover_sum= turnover_sum+turnover[0]
        turnover_avg=turnover_sum/countdays
        if (turnover_avg > threshold1)  and (turnover_avg< threshold2):
            new_stock_list.append(security)
    return new_stock_list


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import marshal
data1 = ['abc',12,23]    #几个测试数据
data2 = {1:'aaa',"b":'dad'}
data3 = (1,2,4)
  
output_file = open("a.txt",'wb')#把这些数据序列化到文件中，注：文件必须以二进制模式打开
marshal.dump(data1,output_file)
marshal.dump(data2,output_file)
marshal.dump(data3,output_file)
output_file.close()
 
 
input_file = open('a.txt','rb')#从文件中读取序列化的数据
#data1 = []
data1 = marshal.load(input_file)
data2 = marshal.load(input_file)
data3 = marshal.load(input_file)
print data1#给同志们打印出结果看看
print data2
print data3
 
 
outstring = marshal.dumps(data1)    #marshal.dumps()返回是一个字节串，该字节串用于写入文件
open('out.txt','wb').write(outstring)
 
 
file_data = open('out.txt','rb').read()
real_data = marshal.loads(file_data)
print real_data
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
data = [1,2,3,0,0,0,3,3,2,3,3,2,3,2,1,2,3]
l = 4
N = [[0] * l for i in range(l)]

Single_amount = []
for i in data:
    print i
    amount += 1
    N[i[0]][i[1]] += 1

print amount
for i in range(l):
    for j in range(l):
        N[i][j] /= float(amount)
print N

perform =[1.6242555495397597, -4.518761204178728, -21.726755218216294, -4.518761204178728, -29.638009049773707, -38.976461655277106, -9.37723693629206, 33.57043719639145, -3.4602076124567427, -19.59324496095875, 10.823529411764788, -0.9583429940490721, -22.369878183831688, 13.350958360872573, -9.205211355748991, -6.741846503326256, -13.334591907368884, -16.255133600365028, -2.3657870791629136, 3.2493907392363526, -13.178294573643425, -6.661737026082243, -16.34755652808168, -1.9367991845056376, 3.413821815154016, -13.131529984256481]
print len(perform)
for i in range(len(Test_Wanted_codes)):
    if perform[i]<-15:
        print(Test_Wanted_codes[i],':',perform[i])

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
  
import tensorflow as tf

def compute_accuracy(v_x, v_y):
    #print 'v_x',type(v_x),v_x.shape
    global prediction
    #input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x:v_x})
    #find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    #calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #get input content
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y})
    return result

def add_layer(inputs, in_size, out_size, activation_function=None,):
    #init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    #calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #add the active hanshu
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
    
if __name__ == '__main__':
    '''
    h_data = ts.get_hist_data(code).sort_index().reset_index()#有时候有些股票会丢失数据
    k_data=ts.get_k_data(code)
    print 'k_data_columns',k_data.columns
    print 'h_data_columns',h_data.columns
    '''
    code = '603603';import data_const as dtcst;import pandas as pd;import numpy as np;import util;
    codata = pd.read_csv(dtcst.My_Database_Dir+code+dtcst.Suffix,index_col='date');print 'codata shape',codata.shape
    m=7;n=15;thresh=25;xy_matrix=np.delete(np.array( [range(len(codata.columns)*m+2)] ), 0, axis=0);
    while(len(codata)>=n+m):
        pc=[1,0] if util.getlast_n_days_p_change(codata, code=None, n=n)>=thresh else [0,1]
        #print util.getlast_n_days_p_change(codata, code=None, n=n),pc
        x_vector=[];
        for i in range(1,m+1):
            x_vector=x_vector+codata.iloc[ -(n+i) ].tolist()
        xy_matrix = np.append(xy_matrix,[x_vector+pc],axis=0)
        codata.drop(index=codata.index[-1],inplace=True)
    print 'xy_matrix',xy_matrix.shape
    print 'y',xy_matrix[:,[-2,-1]].shape,xy_matrix.shape 
    #define placeholder for input
    input_dimens=224;out_put_dimens=2;
    x = tf.placeholder(tf.float32, [None, input_dimens])
    y = tf.placeholder(tf.float32, [None, out_put_dimens])
    #add layer
    prediction = add_layer(x, input_dimens, out_put_dimens, activation_function=tf.nn.softmax)
    #calculate the loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
    #use Gradientdescentoptimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #init session
    sess = tf.Session()
    #init all variables
    sess.run(tf.global_variables_initializer())
    #start training
    for i in range(1):
        #get batch to learn easily
        np.random.shuffle(xy_matrix);
        line=int(len(xy_matrix)*5/6)
        print('xy_matrix',xy_matrix.shape,'line',line)
        batch_x, batch_y = xy_matrix[0:line,0:-2],xy_matrix[0:line,[-2,-1]]
        sess.run(train_step,feed_dict={x: batch_x, y: batch_y})
        if i % 50 == 0 or True:
            print(compute_accuracy( xy_matrix[line:,0:-2],xy_matrix[line:,[-2,-1]] ));
        



