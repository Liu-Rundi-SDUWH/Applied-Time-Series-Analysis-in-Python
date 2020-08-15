'''
Data scraping from the Web: Reading data from HTML tables (web-scraping).
'''
#爬取数据并作图(由于无法访问国外网站，所以选取了一个国内网站做示例)
import re
import requests
from matplotlib import pyplot as plt
# 获取济南近30天的最低温和最高温
html = requests.get('https://www.yangshitianqi.com/jinan/30tian.html').text
#使用正则提取数据
pattern_temperature = r'<div class="fl i3 nz">(\d+~\d+)℃</div>'
pattern_date = r'<div class="t2 nz">(\d\d\.\d\d)</div>'
temperature = re.findall(pattern_temperature, html)
date = re.findall(pattern_date, html)
# 整理数据
max_d = [int(i.split('~')[1]) for i in temperature]
print(max_d)
min_d = [int(i.split('~')[0]) for i in temperature]
print(min_d)
# 定义图像质量
plt.figure(figsize=(9, 4.5), dpi=180)
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 绘制图像
plt.plot(range(30), max_d, linestyle=':')
plt.plot(range(30), min_d, linestyle=':')
# xy轴标识
plt.xlabel('date', size=24)
plt.ylabel('tem/℃', size=24)
plt.title('the latest 30 days in Jinan', size=24)
# 显示网格
plt.grid(axis='y')
# 保存图像
plt.savefig('a.png')
# 显示图像
plt.show()

'''
Read xlsx data and do simple chart operation 
'''
import pandas as pd
import numpy as np
#读取excel数据  dataRH.xlsx
dataRH=pd.read_excel('/home/yinaihua/Desktop/时间序列/期末作业/weihai_pdf/dataRH.xlsx' )
#查看数据集的前6行
print(dataRH.head(6))

#查看读取的数据类型
print(type(dataRH))
#查看变量
#print(dataRH['age'].value_counts())
#绘制柱状图
plt.xlabel("age")
plt.ylabel("Number")
plt.xticks(np.arange(len(dataRH["age"].value_counts()))+0.5,dataRH["age"].value_counts().index)
plt.bar(np.arange(len(dataRH["age"].value_counts()))+0.5,dataRH["age"].value_counts())
plt.savefig('Age.png')  #保存图片
plt.show()

#考虑两个变量来构建一个双变量表(!!!!!!!!!必须得是列联表)
dataRH_2 = pd.crosstab(dataRH['sex'],dataRH['wa'])
print(dataRH_2)
#练习：通过将变量切成不同的切片（片段）来修改变量

'''
Testing the difference between two means (quantitative variables)
'''
import numpy as np
import pandas as pd
from scipy import stats
titanic = pd.read_excel('dataRH.xlsx')
# print(titanic.head(10))

table1 = pd.crosstab(titanic['sex'],titanic['jobtitle'],margins=False)
#交叉表，用于统计两个变量之间的数据个数。

print(table1)
result = stats.chi2_contingency(table1)#卡方检验函数
print(result)

# Consider this example
data1 = np.random.randn(250,2)
#print(data1)
table1 = pd.crosstab(data1[:,0],data1[:,1],margins=False)#交叉表，用于统计两个变量之间的数据个数。

result = stats.chi2_contingency(table1)#卡方检验函数
print(result)

'''
Are you able to interpret it? Both the way I generate the data and the results?
'''
data2 = np.random.randint(5,size=500)
#print(data2)
data3 = np.ones(500)
data4 = np.add(data2,data3)
#print(data4)
table1 = pd.crosstab(data2,data4,margins=False)#交叉表，用于统计两个变量之间的数据个数。
result = stats.chi2_contingency(table1)#卡方检验函数
print(result)

'''
Testing the difference between two means (quantitative variables)
'''
from scipy import stats
data = pd.read_excel('dataRH.xlsx')

data1 = data['wage'][data['sex']==0]
data2 = data['wage'][data['sex']==1]

result = stats.ttest_ind(data1,data2)
print(result)

# Consider another example
import math
from random import gauss

random_numbers = [gauss(1, 2) for i in range(500)]
data1 = [random_numbers[i]*25 +2000 for i in range(500)]
data1 = np.array(data1)
data = data1.reshape(250,2)
#print(data)
result = stats.ttest_ind(data[:,0],data[:,1])
print(result)

'''
Are you able to interpret it? Both the way I generate the data and the results?
'''
random_numbers = [gauss(1, 2) for i in range(500)]
data1 = [random_numbers[i]*20 +2500 for i in range(500)]
data1 = np.array(data1)
data1 = data1.reshape(250,2)

data2 = [random_numbers[i]*15 +2000 for i in range(500)]
data2 = np.array(data2)
data2 = data2.reshape(250,2)
#print(data)
result = stats.ttest_ind(data1,data2)
print(result)
