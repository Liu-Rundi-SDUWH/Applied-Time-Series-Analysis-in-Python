'''
Example 1: GARCH模型-以美元汇率"dollar.xlsx"数据为例
'''

import pandas as pd
from matplotlib import pyplot as plt

data0 = pd.read_excel('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/dollar.xlsx')
df = data0['data']
print(df)

data=(df/df.shift(1)-1).dropna()
plt.figure(figsize=(20,8))
plt.plot(data,'r-')
plt.title('Dollar Dataset')
plt.show()

from arch import arch_model
import numpy as np
garch=arch_model(y=data,mean='Constant',lags=0,vol='GARCH',p=1,o=0,q=1,dist='normal')
garchmodel=garch.fit()
print(garchmodel.params)
garchmodel.plot()
plt.show()

vol=np.sqrt(garchmodel.params[1]/(1-garchmodel.params[2]-garchmodel.params[3]))
print(vol)

'''
Example 2: 预测-接上GARCH模型
'''
u30=data[-30:]
u30=np.matrix(u30)
vol302=np.zeros(30)
vol302[0]=data[-34:-29].std()
for i in range(29):
    vol302[i+1]=np.sqrt(0.1*u30[0,i]**2+0.88*vol302[i]**2)

plt.plot(vol302,label='GARCH(1,1)')
plt.show()

