# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 22:49:39 2016

@author: 현수
"""

import pandas as pd
import random
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')

path= "E:ba\\"

train_ori = pd.read_csv(path+"train.csv", sep=",")
#test = pd.read_csv(path+"test.xlsx", sep=",")

col = train_ori.columns.tolist()

'''
for a in col:
    print(train[a].unique())
''' 
    

drop_col=['Descript', 'Resolution']
train = train_ori.drop(drop_col, axis=1)

from sklearn.cross_validation import train_test_split

random.seed(0)
train, test = train_test_split(train, test_size=0.4)
random.seed(0)
test, valid = train_test_split(test, test_size=0.5)

train.to_csv(path+'train1.csv', sep=',')
test.to_csv(path+'test1.csv', sep=',')
valid.to_csv(path+'valid1.csv', sep=',')


train = pd.read_csv(path+'train1.csv', sep=',')
## Date variable
train['Dates']=pd.to_datetime(train['Dates'])
train['Year']=train['Dates'].dt.year
train['Month']=train['Dates'].dt.month
train['Hour']=train['Dates'].dt.hour
train['Minute']=train['Dates'].dt.minute




###############################Data Exploration

## 카테고리별 카운트
train['Category'].value_counts().plot(kind='barh', legend='reverse', colormap='Accent')

## 년도별 범죄빈도
train['Year'].value_counts().sort_index().plot(kind='bar')

## 월별 범죄빈도
train['Month'].value_counts().sort_index().plot(kind='bar')

## 요일별 범죄빈도
train['DayOfWeek'].value_counts().plot(kind='bar')

## 시간별 범죄빈도
train['Hour'].value_counts().sort_index().plot(kind='bar')





categ_uni=train['Category'].unique().tolist()

## 중범죄만 골라내기
target_list=["ARSON", "ASSAULT","VEHICLE THEFT","ROBBERY", "WEAPON LAWS", "KIDNAPPING",
             "DRUG/NARCOTIC", "TRESPASS", "RUNAWAY", "SEX OFFENSES FORCIBLE"]
non_target_list=[a for a in categ_uni if a not in target_list]
train['Categ']=train['Category'].replace(target_list,1)
train['Categ']=train['Categ'].replace(non_target_list,0)





## 카테고리별 카운트
train['Categ'].value_counts().plot(kind='bar', colormap='Accent')

## 년도별 중범죄, 경범죄 카운트
train.groupby(['Year','Categ']).size().unstack().plot(kind='bar',stacked=True, colormap='Accent')
train.groupby(['Year','Categ']).size().unstack().plot(kind='bar',stacked=False, colormap='Accent')

## 월별 중범죄, 경범죄 카운트
train.groupby(['Month','Categ']).size().unstack().plot(kind='bar',stacked=True, colormap='Accent')
train.groupby(['Month','Categ']).size().unstack().plot(kind='bar',stacked=False, colormap='Accent')
## 요일별 중범죄, 경범죄 카운트
train.groupby(['DayOfWeek','Categ']).size().unstack().plot(kind='bar',stacked=True, colormap='Accent')
train.groupby(['DayOfWeek','Categ']).size().unstack().plot(kind='bar',stacked=False, colormap='Accent')

## 시간별 중범죄, 경범죄 카운트
train.groupby(['Hour','Categ']).size().unstack().plot(kind='bar',stacked=True, colormap='Accent')
train.groupby(['Hour','Categ']).size().unstack().plot(kind='bar',stacked=False, colormap='Accent')


## 시간별 중범죄, 경범죄 카운트
train.groupby(['PdDistrict','Categ']).size().unstack().plot(kind='bar',stacked=True, colormap='Accent')
train.groupby(['PdDistrict','Categ']).size().unstack().plot(kind='bar',stacked=False, colormap='Accent')
















