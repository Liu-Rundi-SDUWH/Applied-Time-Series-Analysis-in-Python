'''
Example 2: An SETAR Model-以太阳黑子”Sunpots"数据集为例
'''
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.api import qqplot
data = sm.datasets.sunspots.load_pandas().data
data.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del data["YEAR"]
data.plot(figsize=(12,8))
plt.title("Sunpots")
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
plt.show()

#AR(2)
arma_mod20 = ARIMA(data, order=(2, 0, 0)).fit()
print(arma_mod20.params)

#AR(3)
arma_mod30 = ARIMA(data, order=(3, 0, 0)).fit()
print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
print(arma_mod30.params)
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)


'''
Example 3: An ESTAR Model-以“Interests_rates.xlsx"数据为例
'''
import pandas as pd
from matplotlib import pyplot as plt

data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/interest_rates.xlsx')
df = data0['data']
print(df)
plt.plot(df,'b-')
plt.title("interest_rates")
plt.show()

data=(df/df.shift(1)-1).dropna()
plt.plot(data,'r-')
plt.title('interest_rates')
plt.show()


'''
Example 4: 马尔科夫模型-以美元汇率“dollar.xlsx"数据为例
'''
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/dollar.xlsx')
df = data0['data']
print(df)
df.plot(title='Dollar', figsize=(12,3))
plt.show()

# 拟合模型
mod_hamilton = sm.tsa.MarkovAutoregression(df, k_regimes=2, order=4, switching_ar=False)
res_hamilton = mod_hamilton.fit()
print(res_hamilton.summary())
