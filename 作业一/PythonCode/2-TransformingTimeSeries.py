'''
1.Distributional transformations
'''
import datetime
import pandas as pd
import tushare as ts  # 该模块是一个免费提供股票交易数据的API<br><br>
import matplotlib.pyplot as plt
import numpy as np
# 我们将看看从2016年1月1日开始过去一年的股票价格
start = datetime.date(2016,1,1)
end = datetime.date.today()

# 得到国金证券公司的股票数据；股票代码是600109
# 第一个参数是获取股票数据的股票代码串，第二个参数是开始日期，第三个参数是结束日期
guojin = ts.get_k_data('600109',start='2016-01-01', end='2020-08-05',autype='qfq')
print (guojin)


plt.figure(figsize=(25, 10))
plt.plot(guojin['date'],guojin['close'])
plt.grid()
plt.title("GuoJin close in the last one month")
plt.show()


'''
2.Stationarity inducing transformations
'''
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

#加载数据集
url = "http://archive.ics.uci.edu//ml//machine-learning-databases//wine//wine.data"
raw_data = urllib.request.urlopen(url)
dataset_raw = np.loadtxt(raw_data, delimiter=",")
print(dataset_raw)
print("over")

#选取最后一列的数据作为y值
data = []
for i in range(0,len(dataset_raw)):
    data.append(dataset_raw[i][-1])
print(data)


import matplotlib.pyplot as plt
x = [i for i in range(0,len(data))]
y = data

plt.plot(x, y, ls="-", lw=2, label="white data")
plt.title("white data")
plt.legend()
plt.show()


#first-diffenence
for i in range(0,len(data)-1):
    data[i] = data[i+1] -  data[i]
x = [i for i in range(0,len(data))]
y = data

plt.plot(x, y, ls="-", lw=2, label="first-diffenence")
plt.title("first-diffenence")
plt.legend()
plt.show()


#second-diffenence
for i in range(0,len(data)-1):
    data[i] = data[i+1] -  data[i]
x = [i for i in range(0,len(data))]
y = data

plt.plot(x, y, ls="-", lw=2, label="second-diffenence")
plt.title("second-diffenence")
plt.legend()
plt.show()


'''
3. Decomposing a time series and smoothing transformations
'''
import pandas as pd
import matplotlib.pylab as plt
data = pd.read_excel("C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/天然气近三年数据.xls")
print(data)
data = data.fillna(method='pad')

plt.figure(figsize=(25, 10))
plt.plot(data['指标'],data['天然气产量当期值(亿立方米)'],'b-',lw = 2)
plt.plot(data['指标'],data['天然气产量累计值(亿立方米)'],'r-')
plt.plot(data['指标'],data['天然气产量同比增长(%)'],'g:')
plt.title("Trend of natural gas in China in recent three years")

plt.legend(["Current value of natural gas production",
            "Cumulative value of natural gas production",
            "Year on year growth of natural gas production(%"])
plt.show()
