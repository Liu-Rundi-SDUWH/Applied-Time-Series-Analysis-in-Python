import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings

# 获取数据
data0 = pd.read_csv('./BAC.csv')
data=data0['BAC.Open']
train = data[0:3000]
test = data[3000:3005]
train.plot(title= 'GOOGL_train')
plt.show()

# 分解
decomposition = seasonal_decompose(train, freq=12)
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(12, 6)
plt.show()

# 判断稳定性
def test_stationary(train):
    # Determing rolling statistics
    rolmean = train.rolling(window=12).mean()
    rolstd = train.rolling(window=12).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(train, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(train, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationary(train)

# 一阶差分
first_difference = train.diff(1)
test_stationary(first_difference.dropna(inplace=False))

# 确定阶数
pmax = 6
qmax = 6
# aic矩阵
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        # 存在部分报错，所以用try来跳过报错。
        try:
            print(ARIMA(train, (p, 1, q)).fit().bic)
            tmp.append(ARIMA(train, (p, 1, q)).fit().bic)
        except:
            tmp.append(100000)
    bic_matrix.append(tmp)

# 从中可以找出最小值
bic_matrix = pd.DataFrame(bic_matrix)
p, q = bic_matrix.stack().idxmin()
print(u'BIC最小的p值和q值为：%s、%s' % (p, q))


# 确定季节参数
p = d = q = range(0, 3)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
score_aic = 1000000.0
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param_seasonal in seasonal_pdq:
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=[5,1,2],
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print('x{}12 - AIC:{}'.format(param_seasonal, results.aic))
    if results.aic < score_aic:
        score_aic = results.aic
        params = param_seasonal, results.aic
param_seasonal, results.aic = params
print('x{}12 - AIC:{}'.format(param_seasonal, results.aic))

# 模型建立
mod = sm.tsa.statespace.SARIMAX(train,
                                order=(5, 1, 2),
                                seasonal_order=(2, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()
print(mod.summary())


# 模型评估
mod.plot_diagnostics(figsize=(15, 12))
plt.title("Test")
plt.show()

# 预测
predictions = mod.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False,tpy='levels')######有差分，要加typ='levels'
predictions=np.matrix(predictions)

test=np.matrix(test)
predictions=np.matrix(predictions)
print(test)
print(predictions)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
