import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabaz_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv

data_ori = pd.read_csv('.\heros.csv',encoding = 'gb18030')
data_ori2 = pd.read_csv('.\heros.csv',encoding = 'gb18030')

print(data_ori.info())
print(data_ori.describe())
print(data_ori.head())
print(data_ori.tail())

data2 = data_ori.iloc[:,1:-2]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
corr = data2.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)
plt.show()

features_remain = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻', u'最大物防', u'初始物防', u'最大每5秒回血', u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features_remain]
data[u'最大攻速'] = data[u'最大攻速'].apply(lambda x: float(x.strip('%'))/100)
data[u'攻击范围']=data[u'攻击范围'].map({'远程':1,'近战':0})

data2[u'最大攻速'] = data2[u'最大攻速'].apply(lambda x: float(x.strip('%'))/100)
data2[u'攻击范围']=data2[u'攻击范围'].map({'远程':1,'近战':0})

ss = StandardScaler()
data = ss.fit_transform(data)
data2 = ss.fit_transform(data2)

gmm = GaussianMixture(n_components=30)
gmm.fit(data)

prediction = gmm.predict(data)

data_ori.insert(0, '分组', prediction)
data_ori.to_csv('./hero_out.csv', index=False, sep=',')


gmm2 = GaussianMixture(n_components=30)
gmm2.fit(data2)
prediction2 = gmm.predict(data2)
data_ori2.insert(0, '分组', prediction2)
data_ori2.to_csv('./hero_out2.csv', index=False, sep=',')

gmm3 = GaussianMixture(n_components=3)
gmm3.fit(data)
prediction3 = gmm3.predict(data)

for i in range(2,31):
    print(i)
    gmmx = GaussianMixture(n_components= i )
    gmmx.fit(data)
    predictionx = gmmx.predict(data)
    print(calinski_harabaz_score(data,predictionx))


print(calinski_harabaz_score(data,prediction))
print(calinski_harabaz_score(data2,prediction2))
print(calinski_harabaz_score(data,prediction3))