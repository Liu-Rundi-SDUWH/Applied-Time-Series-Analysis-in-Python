import numpy as np
import pandas as pd
import re
import requests
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import chi2

'''
Example 1：寻找合适的ARIMA模型并进行评估-以“rainfall.xlsx”数据为例
'''
data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/rainfall.xlsx')
data = data0['data'][1:500]
print(data)

def adf_test(ts):
    from statsmodels.tsa.stattools import adfuller
    adftest = adfuller(ts)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])

    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res

#####定义数据直观分析函数
def SimAnlysis_ts(ts, w):
    roll_mean = ts.rolling(window=w).mean()  # w个数据为一个窗口，取均值
    roll_std = ts.rolling(window=w).std()  # 取方差
    pd_ewma = ts.ewm(span=w).mean()  # # 指数平均线。ts：数据；span：时间间隔

    plt.clf()
    plt.figure()
    plt.grid()
    plt.plot(ts, color='blue', label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='black', label='Rolling Std')
    plt.plot(pd_ewma, color='yellow', label='EWMA')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')  ##滑动平均值和标准偏差
    plt.show()

SimAnlysis_ts(data, 10)


#####定义差分效果函数
def difference_x(data, k):
    plt.plot(data.diff(k).dropna(), 'g-')
    print('k阶差分效果')
    print(adf_test(data.diff(k).dropna()))


# 如果p值显著大于0.05，统计量化大于三个或者两个水平值，差距越大，越不平稳。
# 若统计量显著小于三个置信度且p值接近0，为平稳序列
# 其他情况，可能是数据量不够的原因没有展现趋势
difference_x(data, 1)


def set_pq(D_data):
    from statsmodels.tsa.arima_model import ARIMA
    pmax = 6
    qmax = 6

    # bic矩阵
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            # 存在部分报错，所以用try来跳过报错。
            try:
                print(ARIMA(D_data, (p, 1, q)).fit().bic)
                tmp.append(ARIMA(D_data, (p, 1, q)).fit().bic)
            except:
                tmp.append(100000)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
# set_pq(data)


from statsmodels.tsa.arima_model import ARIMA

model_8_1 = ARIMA(data, (2, 1, 3)).fit()
print(model_8_1.summary2())  # 给出一份模型报告

# 预测结果进行评估
test_81 = data0['data'][500:505]
predictions__81 = model_8_1.predict(start=len(data),
                                    end=len(data) + len(test_81) - 1, dynamic=False,
                                    typ='levels')
predictions__81 = np.matrix(predictions__81)
test_81 = np.matrix(test_81)

for i in range(5):
    # print(test[i])
    print('predicted_81=%f, expected=%f', predictions__81[0, i], test_81[0, i])

from sklearn.metrics import mean_squared_error
error = mean_squared_error(test_81, predictions__81)
print('Test_81 MSE: %.3f' % error)

'''
Example 2：HP filter-以“macrodata"数据集为例
'''
import statsmodels.api as sm
nt=sm.datasets.macrodata.NOTE

df=sm.datasets.macrodata.load_pandas().data
print(df.head())

index=pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
df.index=index

gdp_cyclical, gdp_trend=sm.tsa.filters.hpfilter(df['realgdp'])
print(gdp_trend.head())

df['gdp_trend']=gdp_trend
df['gdp_cyclical']=gdp_cyclical
df[['realgdp', 'gdp_trend']].plot(figsize=(10, 5), color=['black', 'yellow'], linestyle='dashed')
plt.show()

'''
Example 3：HP filter(续）-以“rainfall.xlsx"数据集为例
'''
cyclical, trend=sm.tsa.filters.hpfilter(data)
plt.plot(trend,'b-.')
plt.plot(data,'c-.')
plt.show()
