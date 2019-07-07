import pandas as pd
import numpy as np
from pandas import DataFrame,Series
s = pd.Series([i*2 for i in range(1,11)])
dates = pd.date_range('20170301',periods=8)
df = pd.DataFrame(np.random.randn(8,5),index=dates,columns=list('ABCDE'))

print(df.sort_values('C',ascending=False))  ##对C降序排序
print(df.sort_index(axis=0,ascending=False))  #axis=0表示对纵轴的列进行排序，axis=1表示对横轴进行排序
print(df.describe())

##基于索引的选取用loc
print(df.loc['20170301':'20170302',['B','D']])  ##选取部分数据
print(df.at[dates[0],'C'])   ##选取某一个值
##基于下标的选取用iloc
print(df.iloc[0:3,2:4])

##设置
sl = pd.Series(list(range(10,18)),index=pd.date_range('20170301',periods=8))
df['F'] = sl
df.iat[1,1]=1

##缺失值处理
df.loc[:,'D'] = np.array([4]*len(df))
df1 = df.reindex(index=dates[:4],columns=list('ABCD')+['H'])
df1.loc[dates[0]:dates[1],'G']=1
del df1['H']
print(df1.dropna())
print(df1.fillna(value=0))

##表格拼
s = pd.Series([1,2,4,np.nan,5,7,9,10],index=dates)
print(s.shift(2)) ##循环移动
print(s.diff())
print(s.value_counts())
print(df.apply(np.cumsum))  ##df累加
print(df['A'].cumsum())  #A列累加
print(df.apply(lambda x:x.max()-x.min()))  ##自定义每列极差

##拼接
del df['G']
pieces = [df[:3],df[-3:]]
print(pd.concat(pieces))
left = pd.DataFrame({'key':['x','y'],'value':[1,2]})
right = pd.DataFrame({'key':['x','Z'],'value':[3,4]})
print('LEFT:',left)
print('RIGHT:',right)
print(pd.merge(left,right,on='key',how='inner'))  #left outer inner/default

df3 = pd.DataFrame({'A':['a','b','c','d'],'B':list(range(4))})
print(df3.groupby('A').sum())

##reshape
import datetime
df4 = pd.DataFrame({'A':['one','two','theree','four']*6,
                    'B':['a','b','c']*8,
                    'C':['foo','foo','foo','bar','bar','bar'] *4,
                    'D':np.random.randn(24),
                    'E':np.random.randn(24),
                    'F':[datetime.datetime(2017,i,1) for i in range(1,13)]+
                        [datetime.datetime(2017,i,15) for i in range(1,13)]})
print(pd.pivot_table(df4,values='D',index=['A','B'],columns=['C']))   ##交叉表

##时间序列
t_exam = pd.date_range('20170301',periods=10,freq='S')
print((t_exam))

##读取保存文件
df6 = pd.read_csv('test.csv')
df7 = pd.read_excel('test.xlsx','Sheet1')

df6.to_csv('test2.csv')
df7.to_excel('test2.xlsx')