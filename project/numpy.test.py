import numpy as np
lst=[[1,3,5],[2,4,6]]
print(type(lst))
np_lst=np.array(lst)
print(type(np_lst))
np_lst=np.array(lst,dtype=np.float)
print(np_lst.shape)
print(np_lst.ndim)
print(np.zeros([2,4]))
print("Rand:")
print(np.random.rand(2,4))
lst=np.arange(1,11).reshape([2,5])
print(lst)
print(np.exp(lst))
lst=np.array([[[1,2,3,4],
               [4,5,6,7]],
              [[7,8,9,10],
             [10,11,12,13]],
             [[14,15,16,17],
              [18,19,20,21 ]]])
print(lst)

from numpy.linalg import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
names1880 = pd.read_csv('names/yob1880.txt',names=['name','sex','births'])
years = range(1880,2011)
pieces = []
columns = ['name','sex','births']
for year in years:
	path = 'names/yob%d.txt'%year
	frame = pd.read_csv(path,names=columns)
	frame['year'] = year
	pieces.append(frame)
names = pd.concat(pieces,ignore_index=True)
names
total_births = names.pivot_table('births',index=["year"],columns=["sex"],aggfunc=sum)
total_births.tail()
total_births.plot(title='Total births by sex and year')
def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group
names = names.groupby(['year','sex']).apply(add_prop)
np.allclose(names.groupby(['year','sex']).prop.sum(),1)

##select the first 1000 names by the combination of sex/year##
def get_top1000(group):
    return group.sort_index(by='births',ascending=False)[:1000]

grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)

## analyse of name
boys = top1000[top1000.sex == 'M']
boys = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births',index=["year"],columns=["name"],aggfunc=sum)

subset = total_births[['John','Harry','Mary','Marilyn']]
subset.plot(subplots=True,figsize=(12,10),grid=False,
            title="Number of birth per year")
table = top1000.pivot_table('prop',index=["year"],columns=["sex"],aggfunc=sum)
table.plot(title='Sum of table1000.prop by the year and sex',
           yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))
##choose the 50% point of 1000 names
df = boys[boys.year==2010]
prop_cumsum = df.sort_index(by= 'prop',ascending=False).prop.cumsum()
prop_cumsum[:10]
prop_cumsum.searchsorted(0.5)

##the last letter
get_last_letter = lambda x:x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letters'
names["last_letters"] = last_letters
table = names.pivot_table('births',index=["last_letters"],columns=["sex","year"],aggfunc=sum)
subtable = table.reindex(columns=[1910,1960,2010],level='year')
subtable.head()

import numpy as np
import matplotlib.pylot as plt
import pandas as pd
from pandas import Series,DataFrame

data = np.random.normal(0,1,30).cumsum()
plt.figure()
plt.plot(data,'k*-',label='Default')
plt.plot(data,'r-',drawstyle='steps-post',label='steps-post')
plt.legend(loc='best')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum())
ticks = ax.set_xticks([0,250,500,750,1000])
labels = ax.set_xticklabels(['one','two','three','four','five'],rotation=30,fontsize='small')
plt.yticks([-10,0,10],['h','j','k'],rotation=0,color='r')
plt.ylabel('xx',color='g',fontsize=14,rotation=20)
plt.title('oh,you are so beautiful')

ax.annotate('yes',xy=(667.88,-23.3617),xytext=(+10,+30),textcoords='offset points',
fontsize=16,arrowprops=dict(arrowstyle='->'))  ##annotate

fig = plt.figure()
axes = plt.subplots(2,1)
data = Series(np.random.rand(16),index=list('abcdefghijklmnop'))
#data.plot(kind='bar',ax=axes[0],color='k',alpha=0.7)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
plt.sca(ax1) ##figure on the first
data.plot(kind='bar',color='k',alpha=0.7)
plt.sca(ax2)
data.plot(kind='barh',color='g',alpha=0.7)


tips = pd.read_csv('names/yob1880.txt',names=['name','sex','birth'])


df = pd.DataFrame(np.random.randn(6,4),index=['one','two','three','four','five','six'],
                  columns=pd.Index(['A','B','C','D'],names='haah'))

