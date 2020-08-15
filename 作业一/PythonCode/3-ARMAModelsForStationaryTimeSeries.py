'''
Example1: 数据-东方航空2010—2019股票数据）
'''
# 1）获取数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/东方航空.csv', index_col='Date')

#print(data.head())
#print(data.dtypes)

# 2）Date类型转换为时间

data.index = pd.to_datetime(data.index)
#print("Select 2010-01:\n", data['2010-01'])

# 3）数据预处理
# 首先获取收盘数据，并将其翻转下顺序，因为前面的数据截图可以看到，数据是逆序的，所以需要处理下。
ts = data['Close']
ts = ts[::-1]
#print('ts',ts)

# 日收益率
ts_ret = np.diff(ts)
#print('日收益率', ts_ret)

# 对数收益率
ts_log = np.log(ts)
ts_diff = ts_log.diff(1)
ts_diff.dropna(inplace=True)
#print('对数收益率', ts_diff)

# 4）数据展示
register_matplotlib_converters()

plt.figure()
plt.grid()
plt.plot(ts, color='blue', label='Original')
#print('ts',ts)

# 5）单位根检验

from statsmodels.tsa.stattools import adfuller

def adf_test(ts):
    adftest = adfuller(ts)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res

adftest = adf_test(ts)
print('单位根检验',adftest)

# 6）定阶（ACF PACF）
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def draw_acf_pacf(ts, w):
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plot_acf(ts, ax=ax1, lags=w)

    ax2 = fig.add_subplot(212)
    plot_pacf(ts, ax=ax2, lags=w)

    plt.show()

draw_acf_pacf(ts, 27)

'''
Example 2: AR和MA模型对比-以东方航空股票数据为例
'''
import pandas as pd
import matplotlib.pyplot as plt

# 1)获取数据
data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/东方航空.csv', index_col='Date')
data.index = pd.to_datetime(data.index)
ts = data['Close']

#print(ts.axes)

# 2）AR模型
from statsmodels.tsa.arima_model import ARMA, ARIMA
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

def draw_ar(ts, w):
    arma = ARMA(ts, order=(w, 0)).fit(disp=-1)
    # ts_predict = arma.predict('2016', '2019', dynamic=True)
    ts_predict = arma.predict()
    plt.clf()
    plt.plot(ts_predict['2016':'2019'], 'r:',label="PDT")
    plt.plot(ts['2010':'2015'], '-',label="ORG")
    plt.legend(loc="best")
    plt.title("AR Test %s" % w)

    plt.show()

draw_ar(ts,4)

# 3）MA模型
def draw_ma(ts, w):
    ma = ARMA(ts, order=(0, w)).fit(disp = -1)
    # ts_predict_ma = ma.predict('2016', '2019', dynamic=True)
    ts_predict_ma = ma.predict()

    plt.clf()
    plt.plot(ts['2010':'2015'], label="ORG")
    # plt.plot(ts_predict_ma)
    plt.plot(ts_predict_ma['2016':'2019'], ':',label="PDT")
    plt.legend(loc="best")
    plt.title("MA Test %s" % w)
    plt.show()

    return ts_predict_ma
draw_ma(ts,1)

# 4)ARMA模型
#ARMA模型
ts.describe()

#%%

from statsmodels.tsa.arima_model import ARMA
from itertools import product

# 设置p阶，q阶范围
# product p,q的所有组合
# 设置最好的aic为无穷大
# 对范围内的p,q阶进行模型训练，得到最优模型
ps = range(0, 6)
qs = range(0, 6)
parameters = product(ps, qs)
parameters_list = list(parameters)

best_aic = float('inf')
results = []
for param in parameters_list:
    try:
        model = ARMA(ts, order=(param[0], param[1])).fit()
    except ValueError:
        print("参数错误：", param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = model.aic
        best_param = param
    results.append([param, model.aic])
results_table = pd.DataFrame(results)
results_table.columns = ['parameters', 'aic']
print("最优模型", best_model.summary())

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.plot(ts['2010':'2015'],label = 'ORG')
arma = ARMA(ts, order=(3, 5)).fit(disp = -1)
ts_predict_arma = arma.predict()
# ts_predict_arma = arma.predict('2016', '2019', dynamic=True)
plt.plot(ts_predict_arma,'r:',label = 'PRE')
plt.plot(ts_predict_arma['2016':'2019'],'r:',label = 'PRE')
plt.title("ARMA(3,5)")
plt.legend()
plt.show()



'''
Example 3: AR模型不同参数对比-以东方航空股票数据为例
'''
import pandas as pd
import matplotlib.pyplot as plt

# 1)获取数据
data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/东方航空.csv', index_col='Date')
data.index = pd.to_datetime(data.index)
ts = data['High']

print(ts)

# 2）AR模型选择阶数
from statsmodels.tsa.arima_model import ARMA
from datetime import datetime
from itertools import product

# 设置p阶，q阶范围
# product p,q的所有组合
# 设置最好的aic为无穷大
# 对范围内的p,q阶进行模型训练，得到最优模型

best_aic = float('inf')
results = []
AIC,BIC = list(), list()
for ps in range(1,13):
    try:
        model = ARMA(ts, order=(ps, 0)).fit()
    except ValueError:
        print("参数错误：", ps)
        continue
    aic = model.aic
    bic = model.bic
    AIC.append(-aic/100)
    BIC.append(-bic/100)
    if aic < best_aic:
        best_model = model
        best_aic = model.aic
        best_param = ps
    results.append([ps, model.aic])
results_table = pd.DataFrame(results)
results_table.columns = ['parameters', 'aic']
# print("最优模型", best_model.summary())

ans = pd.DataFrame({'AIC':AIC,'BIC':BIC})
print(ans)

plt.plot(AIC,'b-',label = "AIC")
plt.plot(BIC,'r-',label = 'BIC')
plt.legend()
plt.show()





