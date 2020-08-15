'''
Example 1: 单位根检验-以东方航空股票数据为例
'''
import pandas as pd
import statsmodels.tsa.stattools as ts
# 1)获取数据
data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/东方航空.csv', index_col='Date')

print(data.head())
print(data.dtypes)

# 2）单位根检验
data = data['Close']
result = ts.adfuller(data)
print(result)

'''
Example 2: 单位根存在性-以鲁商发展股票数据为例
'''
import pandas as pd
import statsmodels.tsa.stattools as ts
# 1)获取数据
data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/鲁商发展.csv', index_col='Date')

print(data.head())
print(data.dtypes)

# 2）单位根检验
data = data['Close']
result = ts.adfuller(data)
print('原始数据',result)

if result[0] < result[4]['1%'] and result[0] < result[4]['5%'] and result[0] < result[4]['10%']:
    print('p值',result[1])
    print('不存在单位根')
else:
    print('存在单位跟')

# 3）一阶差分单位根检验
result = ts.adfuller(data,1)
print('一阶差分',result)
if result[0] < result[4]['1%'] and result[0] < result[4]['5%'] and result[0] < result[4]['10%']:
    print('p值',result[1])
    print('不存在单位根')
else:
    print('存在单位跟')

# 3）二阶差分单位根检验
result = ts.adfuller(data,2)
print('二阶差分',result)
if result[0] < result[4]['1%'] and result[0] < result[4]['5%'] and result[0] < result[4]['10%']:
    print('p值',result[1])
    print('不存在单位根')
else:
    print('存在单位跟')

'''
Example 3: 趋势-以1992-2005年的人口出生率数据为例
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Line_Trend_Model( s, ):
    res = {}
    n = len(s)
    m = 2  # 用于计算估计标准误差，线性趋势方程对应的值为 2
    res['t'] = [(i+1) for i in range(n)]  # 对t进行序号化处理
    avg_t = np.mean(res['t'])
    avg_y = np.mean(s)
    ly = sum( map(lambda x,y : x * y, res['t'], s)) - n * avg_t * avg_y
    lt = sum( map(lambda x : x**2, res['t'])) - n * avg_t ** 2
    res['b'] = ly/lt  # 斜率
    res['a'] = avg_y - res['b'] * avg_t  # 截距
    pre_y = res['a'] + res['b'] * np.array(res['t'])  # 直线趋势线
    res['sigma'] = np.sqrt(sum(map(lambda x,y : (x - y)**2, s, pre_y ))/(n-m))  # 估计的标准误差
    return res

# 引入数据
data = [ 18.24, 18.09, 17.70, 17.12, 16.98, 16.57, 15.64, 14.64, 14.03, 13.38, 12.86, 12.41, 12.29, 12.40,]
dates = pd.date_range('1992-1-1', periods = len(data), freq = 'A')  #'A'参数为每年的最后一天
y = pd.Series( data, index = dates )
# 计算值
param = Line_Trend_Model( y )
pre_y = param['a']+ param['b']* np.array(param['t']) # 趋势值
residual = y - pre_y #残差
db = pd.DataFrame( [param['t'], data, list(pre_y), list(residual),  list(residual**2)],
                    index = [ 't','Y(‰)','Trend','Residual','R sqare'],
                    columns = dates ).T
# 输出结果
print('线性趋势方程拟合过程与结果')
print('='*60)
print(db)
print('='*60)
# 计算预测值
t = 16
yt = param['a']+ param['b']* t
print('2007年人口出生率预测值为 {:.2f}‰'.format(yt))

# 画图
fig = plt.figure( figsize = ( 6, 3 ) )
db['Y(‰)'].plot( style = 'bd-',  label = 'Y' )
db['Trend'].plot( style = 'ro-', label = 'Trend')
plt.legend()
plt.grid(axis = 'y')
plt.title('Trend of Birth Rate(1992~2005)')

plt.show()
#
'''
Example 4：非平稳时间序列与差分-以“Airpassengers"为例
'''
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
#rcParams设定好画布的大小
rcParams['figure.figsize'] = 15, 6

# 1）加载数据
data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/AirPassengers.csv')
# print(data.head())
# print('\n Data types:')
# print(data.dtypes)

# 2）处理数据
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
#---其中parse_dates 表明选择数据中的哪个column作为date-time信息，
#---index_col 告诉pandas以哪个column作为 index
#--- date_parser 使用一个function(本文用lambda表达式代替)，使一个string转换为一个datetime变量
data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
# print(data.head())
# print(data.index)

# 3）判断时序稳定性
def test_stationarity(timeseries):
    # 这里以一年为一个窗口，每一个时间t的值由它前面12个月（包括自己）的均值代替，标准差同理。
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # plot rolling statistics:
    fig = plt.figure()
    fig.add_subplot()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='rolling mean')
    std = plt.plot(rolstd, color='black', label='Rolling standard deviation')

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value

    print(dfoutput)

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
ts = data['#Passengers']
test_stationarity(ts)

# 4）让时序变稳定
ts_log = np.log(ts)
# 移动平均法
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log ,color = 'blue')
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log-moving_avg
ts_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(ts_log_moving_avg_diff)

# halflife的值决定了衰减因子alpha：  alpha = 1 - exp(log(0.5) / halflife)
expweighted_avg = pd.DataFrame.ewm(ts_log,halflife=12).mean()
ts_log_ewma_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff)

# 差分
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

'''
Example 6: 单位根检验-以"Airpassenger"为例
'''
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF:
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()
#Plot PACF:
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

'''
Example 9: 分解-以"Airpassengers"为例
'''
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose(timeseries):
    # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

    return trend, seasonal, residual

#消除了trend 和seasonal之后，只对residual部分作为想要的时序数据进行处理
trend , seasonal, residual = decompose(ts_log)
residual.dropna(inplace=True)
test_stationarity(residual)


# '''
# Example 5: 差分平稳-以济南市GDP（2001~2018）为例
# '''
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
# from statsmodels.tsa.arima_model import ARMA
# # 1)数据绘图
# time_series = pd.Series([1066.20,1200.83,1365.33,1618.87,1876.61,2185.09,2562.81,3017.4,3351.00,3910.50,4406.29,4803.70,5230.19,5770.60,6536.12,7201.96,7856.56])
# time_series.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2017'))
# time_series.plot(figsize=(6,4))
# plt.title("JiNan GDP(2001~2018)")
# plt.show()
#
# ## 时间序列是呈指数形式的，波动性比较大，不是稳定的时间序列；对其取对数，将其转化为线性趋势。
# # 2）转化
# time_series = np.log(time_series)
# time_series.plot(figsize=(6,4))
# plt.title("JiNan GDP_log")
# plt.show()
#
# ## 取对数之后的时间路径图明显具有线性趋势，为了确定其稳定性，对取对数后的数据进行ADF单位根检验
# # 3)单位根检验
# t=sm.tsa.stattools.adfuller(time_series, )
# output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)
#
# ## 由上面的检验可知：t 统计量要大于任何置信度的临界值，因此认为该序列是非平稳的；故对序列进行差分处理，并进行 ADF 检验。
# # 4）差分并进行ADF检验
# time_series = time_series.diff(1)
# time_series = time_series.dropna(how=any)
# time_series.plot(figsize=(8,6))
# plt.title("First-difference")
# plt.show()
# t=sm.tsa.stattools.adfuller(time_series)
# output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)
# ## 差分之后的序列基本达到稳定，且通过了 ADF 检验。

# '''
# Example 7: 差分-以鲁商发展股票数据为例
# '''
# import pandas as pd
# import numpy as np
# import statsmodels.tsa.stattools as ts
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/鲁商发展.csv', index_col='Date')
# # print(data.head())
#
# y = data['Adj Close']
# #一阶差分
# for i in range(0,len(y)-1):
#     y[i] = y[i+1] -  y[i]
# print(y)
#
# x = [i for i in range(0,len(y))]
# plt.plot(x, y, label="first-diffenence")
# plt.title("first-diffenence")
# plt.show()
#
# y = np.array(y)
# result = ts.adfuller(y)
# print(result)
#
# #二阶差分
# for i in range(0,len(y)-1):
#     y[i] = y[i+1] -  y[i]
# print(y)
#
# x = [i for i in range(0,len(y))]
# plt.plot(x, y, label="second-diffenence")
# plt.title("second-diffenence")
# plt.show()
#
# y = np.array(y)
# result = ts.adfuller(y)
# print(result)
#
# '''
# Example 8: 判断-2015/1/1~2015/2/6某餐厅的销售数据
# '''
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.pylab import style
# from statsmodels.tsa.stattools import adfuller as ADF
# from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
# from statsmodels.tsa.arima_model import ARIMA
# import statsmodels.tsa.api as smt
# import seaborn as sns
# style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 参数初始化
# discfile = 'C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/restaurant.xlsx'
# forecastnum = 5
#
# # 读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
# data = pd.read_excel(discfile, index_col=u'日期')
# # print(data)
#
# # 时序图
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# data.plot()
# plt.show()
#
# # 自相关图
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(data).show()
#
# # 平稳性检测
# from statsmodels.tsa.stattools import adfuller as ADF
# print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
# # 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
#
# # 差分后的结果
# D_data = data.diff().dropna()
# D_data.columns = [u'销量差分']
# D_data.plot()  # 时序图
# plt.show()
# plot_acf(D_data).show()  # 自相关图
# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(D_data).show()  # 偏自相关图
# print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))  # 平稳性检测
#
# from statsmodels.stats.diagnostic import acorr_ljungbox
# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值


