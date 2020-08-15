'''
Example 3:  State Space Modeling-以“ global_temps.xlsx”数据集为例
'''

import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm

data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/global_temps.xlsx')
data = data0['data']
data.index= pd.date_range(start='1850-01', end='2018-01', freq='M')
print(data)
plt.plot(data,'c-')
plt.title("Global temperature(1850~2017")
plt.show()

mod = sm.tsa.statespace.SARIMAX(data, order=(2,1,0), seasonal_order=(1,1,0,12), simple_differencing=True)
res = mod.fit(disp=False)
print(res.summary())