import pandas as pd
from matplotlib import pyplot as plt

'''
Example 1: 季节性模型-以“rainfall.xlsx”数据为例
'''
data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/rainfall.xlsx')
data = data0['data']
data = list(data)

data = pd.Series(data, index=pd.date_range('1-1-1959', periods=len(data), freq='M'), name = 'data')
print(data)
print(data.describe())

from statsmodels.tsa.seasonal import seasonal_decompose
stl = seasonal_decompose(data)
fig = stl.plot()
plt.show()


'''
Example 2: 季节性ARIMA-以“俄勒冈州波特兰市平均每月公共汽车载客量数据”为例
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/oregon.csv',
                 parse_dates=['month'], index_col='month')
print(df.head())
df['riders'].plot(figsize=(12, 8), title='Monthly Ridership', fontsize=14)
plt.show()


decomposition = seasonal_decompose(df['riders'], freq=12)
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(12, 6)
plt.show()

#稳定性
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(df['riders'])

#一阶差分
df['first_difference'] = df['riders'].diff(1)
test_stationarity(df['first_difference'].dropna(inplace=False))

#季节差分
df['seasonal_difference'] = df['riders'].diff(12)
test_stationarity(df['seasonal_difference'].dropna(inplace=False))

# 一阶差分+季节差分合并起来
df['seasonal_first_difference'] = df['first_difference'].diff(12)
test_stationarity(df['seasonal_first_difference'].dropna(inplace=False))

# 取对数
df['riders_log'] = np.log(df['riders'])
test_stationarity(df['riders_log'])


# 在此基础上下一阶差分，去除增长趋势：
df['log_first_difference'] = df['riders_log'].diff(1)
test_stationarity(df['log_first_difference'].dropna(inplace=False))

#对对数进行季节性差分：
df['log_seasonal_difference'] = df['riders_log'].diff(12)
test_stationarity(df['log_seasonal_difference'].dropna(inplace=False))


#对数+一阶差分+季节差分的效果:
df['log_seasonal_first_difference'] = df['log_first_difference'].diff(12)
test_stationarity(df['log_seasonal_first_difference'].dropna(inplace=False))

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['seasonal_first_difference'].iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['seasonal_first_difference'].iloc[13:], lags=40, ax=ax2)
plt.show()

mod = sm.tsa.statespace.SARIMAX(df['riders'], trend='n', order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
results = mod.fit()
print(results.summary())


df['forecast'] = results.predict(start=102, end=114, dynamic=True)
df[['riders', 'forecast']].plot(figsize=(12, 6))
plt.show()

npredict = df['riders']['1982'].shape[0]
fig, ax = plt.subplots(figsize=(12, 6))
npre = 12
ax.set(title='Ridership', xlabel='Date', ylabel='Riders')
ax.plot(df.index[-npredict - npre + 1:], df.ix[-npredict - npre + 1:, 'riders'], 'o', label='Observed')
ax.plot(df.index[-npredict - npre + 1:], df.ix[-npredict - npre + 1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')
plt.show()

start = datetime.datetime.strptime("1982-07-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0, 12)]
future = pd.DataFrame(index=date_list, columns=df.columns)
df = pd.concat([df, future])

df['forecast'] = results.predict(start=114, end=125, dynamic=True)
df[['riders', 'forecast']].ix[-24:].plot(figsize=(12, 8))
plt.show()

'''
Example 3: 简单指数平滑-以“全球气温变化（1850~2016）”数据为例
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/global_temps.xlsx')
data = list(data0['data'])
index= pd.date_range(start='1850-01', end='2018-01', freq='M')
globaltemp = pd.Series(data, index)
print(globaltemp)

ax=globaltemp.plot()
ax.set_xlabel("Year")
ax.set_ylabel("temperature(℃)")
plt.title("Global Temperature from 1850 to 2018.")
plt.show()

#1. fit1不使用自动优化，而是选择显式地为模型提供α=0.2
#2.fit2选择一个α=0.63
#fit3我们允许statsmodels自动找到优化的α对我们有价值。这是推荐的方法(省时省力)
fit1 = SimpleExpSmoothing(globaltemp).fit(smoothing_level=0.2,optimized=False)
fcast1 = fit1.forecast(3).rename(r'$\alpha=0.2$')
fit2 = SimpleExpSmoothing(globaltemp).fit(smoothing_level=0.6,optimized=False)
fcast2 = fit2.forecast(3).rename(r'$\alpha=0.6$')
fit3 = SimpleExpSmoothing(globaltemp).fit()
fcast3 = fit3.forecast(3).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

plt.figure(figsize=(20, 8))
plt.plot(globaltemp, marker='o', color='black')
plt.plot(fit1.fittedvalues, marker='o', color='blue')
line1, = plt.plot(fcast1, marker='o', color='blue')
plt.plot(fit2.fittedvalues, marker='o', color='red')
line2, = plt.plot(fcast2, marker='o', color='red')
plt.plot(fit3.fittedvalues, marker='o', color='green')
line3, = plt.plot(fcast3, marker='o', color='green')
plt.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])
plt.show()


'''
Example 4: HW季节预测模型-以“高铁乘客数量”数据为例
'''
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/9-4HW.csv', nrows=10500)

# Creating train and test set
# Index 10392 marks the end of October 2013
train = df[0:7000]
test = df[7000:]

# 以天为单位生成数据集
df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.index = df['Timestamp']
df = df.resample('D').mean()

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']
train = train.resample('D').mean()

test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
test.index = test['Timestamp']
test = test.resample('D').mean()

train.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
test.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
plt.show()


from statsmodels.tsa.api import ExponentialSmoothing

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Count']),
                            seasonal_periods=7,
                            trend='add',
                            seasonal='add', ).fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['Holt_Winter']))
print(rms)



