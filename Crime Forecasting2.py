# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 02:28:19 2016

@author: 현수
"""

import pandas as pd
import random
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')

path= "E:ba\\"

train = pd.read_csv(path+'train1.csv', sep=',')
val = pd.read_csv(path+'valid1.csv',sep=',')
test = pd.read_csv(path+'test1.csv', sep=',')

len_tr=len(train)
len_va=len(val)

train = train.append(val)
train = train.append(test)

## Date variable
train['Dates']=pd.to_datetime(train['Dates'])
train['Year']=train['Dates'].dt.year
train['Month']=train['Dates'].dt.month
train['Hour']=train['Dates'].dt.hour
train['Minute']=train['Dates'].dt.minute



###############################Data Exploration

categ_uni=train['Category'].unique().tolist()

## 중범죄만 골라내기
target_list=["ARSON", "ASSAULT","VEHICLE THEFT","ROBBERY", 
             "WEAPON LAWS", "KIDNAPPING",
             "DRUG/NARCOTIC", "TRESPASS", "RUNAWAY", 
             "SEX OFFENSES FORCIBLE"]
non_target_list=[a for a in categ_uni if a not in target_list]
train['Categ']=train['Category'].replace(target_list,1)
train['Categ']=train['Categ'].replace(non_target_list,0)

dropCol = ['Dates','Address','Category','Unnamed: 0']
train = train.drop(dropCol, axis=1)



## 더미 만들기

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
x=train
cat_var=['PdDistrict','Month','Hour','Year', 'DayOfWeek']
labelEnc=[]
for var in cat_var:
    cat_val=pd.unique(x[var]).tolist()
    enc1=LabelEncoder()
    enc1.fit(cat_val)
    labelEnc.append(enc1)
    x[var]=enc1.transform(x[var].tolist())

enc2=OneHotEncoder(categorical_features=[x.columns.get_loc(var) for var in cat_var])
enc2.fit(x)
x=enc2.transform(x).toarray()


crime_train = x[:len_tr]
crime_val = x[len_tr:(len_tr+len_va)]
crime_test = x[(len_tr+len_va):]


crime_train=enc2.transform(crime_train).toarray()
train_X = crime_train[:,:-1]
train_y = crime_train[:,-1]

crime_val = enc2.transform(crime_val).toarray()
valid_X = crime_val[:,:-1]
valid_y = crime_val[:,-1]

crime_test = enc2.transform(crime_test).toarray()
test_X = crime_test[:,:-1]
test_y = crime_test[:,-1]

from sklearn.metrics import precision_recall_fscore_support



## aive bayse
from sklearn import naive_bayes
nb=naive_bayes.GaussianNB()
nb.fit(train_X, train_y)

cvs_nb=nb.score(valid_X,valid_y)
pred_nb=nb.predict(valid_X)
NV = precision_recall_fscore_support(valid_y, pred_nb, average='binary')


## SVM
from sklearn import svm
clf=svm.SVC(kernel='rbf').fit(train_X, train_y)

cvs_svm=clf.score(valid_X,valid_y)
pred_svm=clf.predict(valid_X)
SVM = precision_recall_fscore_support(train_y, valid_y, average='binary')

## Logistic
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression().fit(train_X,train_y)

cvs_logistic_over=clf.score(valid_X,valid_y)
pred_logistic_over=clf.predict(valid_X)

Log_Reg = precision_recall_fscore_support(valid_y, pred_logistic_over, average='binary')



## Decision Tree
from sklearn import tree
cvs_dt = []

clf = tree.DecisionTreeClassifier()
clf.fit(train_X, train_y)

cvs_dt=clf.score(valid_X,valid_y)
pred_dt=clf.predict(valid_X)

precision_recall_fscore_support(valid_y, pred_dt, average='binary')
s=[]



## KNN
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=700)
neigh.fit(train_X, train_y)

cvs_knn=neigh.score(valid_X,valid_y)
pred_knn=neigh.predict(valid_X)
KNN = precision_recall_fscore_support(train_y, valid_y, average='binary')




## One class learning
x=train_X[train_y==1,:]
from sklearn import svm
#nu=[0.1, 0.3, 0.5]
nu=0.1
#g=[1, 10, 100]
g=1
svdd=svm.OneClassSVM(kernel='rbf',gamma=g,nu=nu, max_iter=300)
svdd.fit(x)
prd=svdd.predict(valid_X)
prd1=[]
for a in prd:
    if a==-1:
        a=0
        prd1.append(a)
    else:
        prd1.append(a)

from sklearn.metrics import accuracy_score
accuracy_score(valid_y, prd1)
precision_recall_fscore_support(valid_y, prd1, average='binary')




## Final Perfomance on test set
score_final=clf.score(test_X, test_y)
pred_final=clf.predict(test_X)
prf_final=precision_recall_fscore_support(test_y, pred_final, average='binary')

