#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tushare as ts
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    df = ts.get_k_data("601398")
    print df
    # Use the prior two days of returns as predictor 
    # values, with direction as the response
    X = df[["volume","high"]]
    y = df["close"]
    X_train, X_test, y_train ,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    #X_train = X_train[:, None]
    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()), 
              ("LDA", LDA()), 
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
                C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
                n_estimators=1000, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0)
              )]

    # Iterate through the models
    for m in models:

        # Train each of the models on the training set
        m[1].fit(X_train, y_train)

        # Make an array of predictions on the ml_stock set
        pred = m[1].predict(X_test)

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        print(pred, y_test)
        #print confusion_matrix(y_test,pred)
        

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import tushare as ts
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import cross_val_score
from ml_stock.LinearRegression_Predictor import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2016, 11, 20)
#从互联网获取数据
df = ts.get_k_data("601398")
#print df
#print(df.head())
df = df[['open',  'high',  'low',  'close', 'volume']]
df['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0
df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]
#print(df.head())
#print(len(df))
forecast_col = 'close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))#.ceil--取整
#预测forecast_out天后的
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.shape)
#print(df.tail()) #默认是5行
X = np.array(df.drop(['label'], 1))


X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True) #删除确实数据
#print(X)
#print(X_lately)
y = np.array(df['label'])
#print(y)
#print(X.shape)
#print(y.shape)
X_train, X_test, y_train ,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy , 'tp / (tp + fp)')

forecast_set = clf.predict(X_lately)

#print(forecast_set,accuracy,forecast_out)

style.use('ggplot')

df['Forecast']=np.nan

df['close'].plot()
#df['Forecast'].plot()
plt.show()


svm
for k in ['linear','poly','rbf','sigmoid']:
    clf2 = svm.SVR(k)
    clf2.fit(X_train,y_train)
    accuracy2 = clf2.score(X_test,y_test)    
    print(accuracy2)
'''
clf3 = svm.SVC(kernel='linear',C=1)
scores = cross_val_score(clf3,X,y,cv=5,scoring='f1_macro')
print(scores)
'''        
        
        
        