#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os,platform
import pandas as pd
from urllib2 import urlopen,Request

#结合最新可以查到实时数据的股票代码，给出缩减后面的codes
def get_wanted_codes():
    import data_const
    data = pd.DataFrame(pd.read_csv(data_const.My_Store_Dir+"realtime_data"+data_const.Suffix))
    #data = ts.get_today_all()
    int_list = data['code'].tolist()
    str_list = [str(item) for item in int_list]
    filled_str_list = intstr_list_zfill(str_list, data_const.HS_stock_code_len)
    tmp = [val for val in filled_str_list if val in data_const.HS_stock_codes]
    data = pd.DataFrame(pd.read_csv(data_const.My_Store_Dir+"basic"+data_const.Suffix))
    #data = ts.get_today_all()
    int_list = data['code'].tolist()
    str_list = [str(item) for item in int_list]
    filled_str_list = intstr_list_zfill(str_list, data_const.HS_stock_code_len)
    tmp = [val for val in filled_str_list if val in tmp]
    return tmp
    
#字符串列表str_list 填充到指定位数zfill_len
def intstr_list_zfill(str_list,zfill_len):
    return [item.zfill(zfill_len) for item in str_list]

def ping_API(url):         
    try:                        
        req = Request(url)                        
        text = urlopen(req,timeout=10).read()
        if len(text) < 15:                                    
            raise IOError('no data!')                   
    except Exception as e:                        
        print(e)            
    else:                        
        return text

"""    功能：                    
    验证输入的股票代码是否正确  
"""   
def is_hs_stock_code(code):  
    import data_const                                
    if len(code) != data_const.HS_stock_code_len:                                    
        return True                       
    else: return False
    
def type_data(object):
    print type(object),':',object
    return object
    
def getSeparator():
    if 'Windows' in platform.system():
        separator = '\\'
    else:
        separator = '/'
    return separator

def findPath(file='Project_Root_File.ini'):
    # all_projects_root_path = os.path.dirname(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])+'/'#可以得到工作空间路径
    # Python_project_root = (all_projects_root_path + 'SAP_Python_0/')
    # return Python_project_root
    #下面是通过放置文件来定位工作路径
    o_path = os.getcwd()
    print 'os.getcwd',o_path
    separator = getSeparator()
    str = o_path
    str = str.split(separator)
    while len(str) > 0:
        spath = separator.join(str)+separator+file
        leng = len(str)
        if os.path.exists(spath):
            spath = spath.strip(file)
            print 'Project_Root_Dir',spath
            return spath
        str.remove(str[leng-1])
#判断是否为数字
def isNum(value):
    try:
        value + 1
    except TypeError:
        return False
    else:
        return True
def time_coefficient(last_exchange_day):
    if getDatetimeToday().strftime("%Y-%m-%d")!=last_exchange_day:time_coefficient = 1
    else:
        trade_time = 4*60#分钟
        already_trade_time=0
        import datetime
        #date = (datetime.datetime.now()-datetime.timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hour = int(date[11:13])
        minue = int(date[14:16])
        #print hour,"minue: ", minue
        if 9<=hour<13:#早上交易时间
            already_trade_time=(hour-9)*60+minue-30
        elif 13<=hour<15:
            already_trade_time = trade_time/2+(hour-13)*60+minue
        else:already_trade_time=trade_time;
        time_coefficient = float(already_trade_time)/trade_time
    #print 'time_coefficient',time_coefficient
    return time_coefficient
        
def not_want_codes_list():
    data1=[]
    try:
        import marshal
        input_file = open(findPath('Project_Root_File.ini')+'not_like_codes_list','rb')#从文件中读取序列化的数据
        #data1 = []
        data1 = marshal.load(input_file)
    except Exception,e:
        print e
        print 'util not_want_codes_list error'
    import data_const as dtcst
    print 'not_want_codes_list',len(data1),'/',len(dtcst.Whole_codes)
    return data1
def code_func_map(code_func_map_exist=None):
    import marshal
    if code_func_map_exist!=None:
        try:
            output_file = open(findPath('Project_Root_File.ini')+'code_func_map','wb')#把这些数据序列化到文件中，注：文件必须以二进制模式打开
            marshal.dump(code_func_map_exist,output_file)
            output_file.close()
            print '\ncode_func_map write over',len(code_func_map_exist),'length'
        except:
            print 'code_func_map 持久化 失败'  
        return
    data1=None
    try:
        input_file = open(findPath('Project_Root_File.ini')+'code_func_map','rb')#从文件中读取序列化的数据
        data1 = marshal.load(input_file)
    except Exception,e:
        print e
        print 'util code_func_map error'
        import ml_stock.not_want_codes_list_object_store as maker
        maker.not_want_codes_list_and_code_func_map_generator(not_want_codes_list_needing=False)
        return code_func_map()
    #print 'util -- code_func_map elements example',data1['000001']
    return data1
def get_codata(path,code):
    return pd.read_csv(path+code+'.csv').set_index('date');
def and_my(p1,p2):
    return p1 and p2
def or_my(p1,p2):
    return p1 or p2
def identity_func(p1=None,p2=True):#幺元函数：类比列表求和，sum = sum+a;以及函数式编程思想
    return p2
def  xor(a,b):# exclusive or;异或逻辑
    return  (a and (not b) ) or ( (not a) and b)
def wierd_abnormal(p1,p2):#反常怪异逻辑
    return not True in [p1,p2]
aol_func_map ={'and':and_my,'or':or_my,'xor':xor,'identity_func':identity_func}
myflatten = lambda x: [y for l in x for y in myflatten(l)] if type(x) is list else [x]
x2Code = lambda x:str(x)+'.SZ' if int(x)<600000 else str(x)+'.SH';
#近n日大盘指标涨跌幅度
def getlast_n_days_p_change(codata=None,code='sz50',n=55,end=0):
    if n<end:raise RuntimeError('getlast_n_days_p_change num<end')
    try:
        if codata==None:#小心dataframe 数据结构似乎对==重载了，和None做比较会出现异常
            import data_const as dtcst
            codata = pd.read_csv(dtcst.My_Store_Dir+code+dtcst.Suffix)
    except: pass
    last_n_close = codata.tail(n).head(n-end)['close']
    last_n_days_p_change = (last_n_close.tail(1).item()-last_n_close.head(1).item())/last_n_close.head(1).item()*100
    return last_n_days_p_change
def get_coherence_date(review_num,review_end):
    import data_const as dtcst
    env_sh_data = get_codata(dtcst.My_Store_Dir,'sh');
    try:
        review_num_date = env_sh_data.tail(review_num).head(review_num-review_end).head(1).index.item()
        review_end_date = env_sh_data.tail(review_num).head(review_num-review_end).tail(1).index.item()
        return review_num_date,review_end_date
    except:
        print ' env_sh_data key date error'
        raise RuntimeError('util.get_coherence_date error')
def review_num_end_mapping(codata,review_num_date_or_wanted,review_end_date_limited,which_want_or_change_direction=1):
    try:
        df = codata;
        review_num =df.tail(1).index[0] - df[(df.date==review_num_date_or_wanted)].index.tolist()[0]
        return review_num,review_num_date_or_wanted
    except:
        if review_num_date_or_wanted == review_end_date_limited:return None,review_num_date_or_wanted
        print ' codata key date to index error',review_num_date_or_wanted,review_end_date_limited,which_want_or_change_direction
        from data import update_data
        review_num_date_or_wanted = getTomorrowDate(review_num_date_or_wanted, d_change=which_want_or_change_direction)
        return  review_num_end_mapping(codata,review_num_date_or_wanted,review_end_date_limited,which_want_or_change_direction)
#返回间隔d_change 的时间，负数为历史时间，正数为未来时间
def get_d_change_Date(date_,d_change=1):
    from datetime import datetime, date, timedelta
    dt = datetime.strptime(date_,'%Y-%m-%d') #date为str再转datetime
    return (dt + timedelta(days = +d_change)).strftime("%Y-%m-%d") 
def vector_cos(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    #print(result1)
    #print(result2)
    #print(result3)
    cos = result1/((result2*result3)**0.5)
    #print 'x:',x,'y:',y
    #print("result is "+str(cos))
    return cos
def interval_mapping(old_list,max_w=1.00000,min_w=-1.0000):
    value=old_list;interval_ = max_w-min_w;
    max_v=max(value);min_v=min(value)
    k = interval_/(max_v-min_v+0.0000)
    transform_value=[k*(x-min_v)+min_w for x in value]
    return transform_value
#得到最近交易日期，并且得到新的附属数据
'''把时间戳转化为时间: 1479264792 to 2016-11-16 10:53:12'''
def TimeStampToTime(timestamp):
    import time
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d %H:%M:%S',timeStruct)
def get_FileModifyTime(filePath):
    import os;import time
    filePath = unicode(filePath,'utf8')
    t = os.path.getmtime(filePath)
    return TimeStampToTime(t),time.strftime('%Y-%m-%d',time.localtime(t))
#获取datetime.datetime类型前一天日期
def getDatetimeYesterday():
    from datetime import datetime, date, timedelta
    today = getDatetimeToday() #datetime类型当前日期
    yesterday = today + timedelta(days = -1) #减去一天
    return yesterday

def getYesterdayDate():
    from datetime import datetime, date, timedelta
    return str(date.today()+ timedelta(days = -1))
def getTomorrowDate(today_date,d_change=1):
    from datetime import datetime, date, timedelta
    dt = datetime.strptime(today_date,'%Y-%m-%d') #date为str再转datetime
    print dt
    return (dt + timedelta(days = +d_change)).strftime("%Y-%m-%d") #明天
def getDatetimeToday():
    from datetime import datetime, date, timedelta
    t = date.today()  #date类型
    dt = datetime.strptime(str(t),'%Y-%m-%d') #date转str再转datetime
    return dt
def midlelize(Y):#中心化 均值化数据
    import numpy as np;Y=np.array(Y);return Y-Y.mean();
def get_r2_numpy(y_fit, y): #获得R square 拟合优度估计量
    if len(y_fit)!=len(y):raise;
    import numpy as np;y_fit=np.array(y_fit);y=np.array(y)
    r_squared = 1 - (sum((y - y_fit)**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return r_squared
def get_pearson_corrcoef(y_fit, y): #皮尔逊相关系数(Pearson Correlation Coefficient) 也可以作为拟合优度估计量
    if len(y_fit)!=len(y):raise;
    import numpy as np;y_fit=np.array(y_fit);y=np.array(y)
    return np.corrcoef(y_fit,y)[0][1]#注意范围是-1到+1 正负相关
def get_first_order_temporal_correlation_coefficient(y_fit,y0):
    if len(y_fit)!=len(y0):raise;
    import numpy as np;y_fit=np.array(y_fit);y=np.array(y0);
    diff_y_fit=np.diff(y_fit);diff_y0=np.diff(y0);
    sum_diff1_mul_diff2= sum(diff_y_fit*diff_y0);sum_diff1_mul_self=sum(diff_y_fit*diff_y_fit);sum_diff2_mul_self=sum(diff_y0*diff_y0);
    return sum_diff1_mul_diff2/np.sqrt(sum_diff1_mul_self*sum_diff2_mul_self) #注意范围是-1到+1 正负相关
dot_mult = lambda (x,y):x*y;
def weight_sum(weight=[],value=[]):
    return sum(map(dot_mult,zip(weight,value)));
def up20(nums=[0]):
    for item in nums:
        if item<=0:return False;
    return True;
if __name__ == '__main__':
    print type(x2Code(5)),x2Code('5');
    
    