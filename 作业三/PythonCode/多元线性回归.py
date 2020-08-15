'''
多元线性回归
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# 数据集处理
# data = pd.read_csv("./BAC.csv")#收盘价
# x1 = data['BAC.Close'][0:2997].reset_index(drop=True)
# x2 = data['BAC.Close'][1:2998].reset_index(drop=True)
# x3 = data['BAC.Close'][2:2999].reset_index(drop=True)
# y =  data['BAC.Close'][3:3000].reset_index(drop=True)
#
# dataset = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'y':y})
# dataset.to_csv('yes.csv', sep='\t')

y = pd.read_csv("./BAC_0.csv",usecols=[4])
y_0 = y
plt.plot(y_0[2950:],'c-',label="expected")
x = pd.read_csv("./BAC_0.csv",usecols=[1,2,3])

x_train = x[0:2997]
x_test = x[2997:]
y_train = y[0:2997]
y_test = y[2997:]


model = LinearRegression()

model.fit(x_train,y_train)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b[0])

MSE = 0
y_predict = [0,0,0,0,0]

y_predict[0] = np.dot(b,x_test.loc[2997] + a)[0]
MSE = MSE + (y_test.loc[2997]-y_predict[0])**2


x_test.loc[2998]['x3'] = y_predict[0]
y_predict[1] = np.dot(b,x_test.loc[2998] + a)[0]
MSE = MSE + (y_test.loc[2998]-y_predict[1])**2


x_test.loc[2999]['x2'] = y_predict[0]
x_test.loc[2999]['x3'] = y_predict[1]
y_predict[2] = np.dot(b,x_test.loc[2999] + a)[0]
MSE = MSE + (y_test.loc[2999]-y_predict[2])**2

x_test.loc[3000]['x1'] = y_predict[0]
x_test.loc[3000]['x2'] = y_predict[1]
x_test.loc[3000]['x3'] = y_predict[2]
y_predict[3] = np.dot(b,x_test.loc[3000] + a)[0]
MSE = MSE + (y_test.loc[3000]-y_predict[3])**2

x_test.loc[3001]['x1'] = y_predict[1]
x_test.loc[3001]['x2'] = y_predict[2]
x_test.loc[3001]['x3'] = y_predict[3]
y_predict[4] = np.dot(b,x_test.loc[3001] + a)[0]
MSE = MSE + (y_test.loc[3001]-y_predict[4])**2
print('pre',y_predict)
print('Test MSE: %.3f' % MSE)


y.loc[3002] = y_predict[0]
y.loc[3003] = y_predict[1]
y.loc[3004] = y_predict[2]
y.loc[3005] = y_predict[3]
y.loc[3006] = y_predict[4]

plt.plot(y[3000:],'r:',label="predicted")
plt.legend()
plt.show()