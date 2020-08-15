
import numpy as np
import pandas as pd
import re
import requests
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import chi2

data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/USB.csv')
print(data.head())
print(data.dtypes)

Open=data['USB.Open']
Close=data['USB.Close']


shoupanjia=Close[::-1]
rishouyi = shoupanjia.diff(1)
plt.plot(rishouyi,'r-')
plt.show()

Close_log = np.log(Close)
rishouyi_log = Close_log.diff(1)
rishouyi_log.dropna(inplace=True)

mydata=rishouyi[1:500]
plt.plot(mydata,'r-')
plt.show()

##########首先是ARIMA（1，2，3）模型
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(mydata, (1,2,3)).fit()
print(model.summary2()) #给出一份模型报告

####评估模型预测结果
test=rishouyi[500:505]
predictions = model.predict(start=len(mydata), end=len(mydata)+len(test)-1,
                            dynamic=False,typ='levels')######有差分，要加typ='levels'
predictions=np.matrix(predictions)
test=np.matrix(test)
# print(predictions[0,1])
# print(test[0,1])
#print(test)
for i in range(5):
    #print(test[i])
    print('predicted=%f, expected=%f' ,predictions[0,i], test[0,i])

from sklearn.metrics import mean_squared_error
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

test=np.array(test)
predictions=np.array(predictions)
plt.plot(test[0,:],'r-')
plt.plot(predictions[0,:],'g-')
plt.show()

############接着，使用AR（2）模型
from statsmodels.tsa.arima_model import ARIMA,ARMA
ar_2= ARMA(rishouyi[1:500], order=(2,0)).fit(disp=-1)
print(ar_2.summary2()) #给出一份模型报告

######对AR（2）模型的预测结果进行评估
test=rishouyi[500:505]
predictions_ar2 = ar_2.predict(start=len(mydata), end=len(mydata)+len(test)-1,
                               dynamic=False)
predictions_ar2=np.matrix(predictions_ar2)
test=np.matrix(test)
# print(predictions[0,1])
# print(test[0,1])
#print(test)
for i in range(5):
    #print(test[i])
    print('predicted_ar2=%f, expected=%f' ,predictions_ar2[0,i], test[0,i])

from sklearn.metrics import mean_squared_error
error = mean_squared_error(test, predictions_ar2)
print('Test MSE: %.3f' % error)

#%%

test=np.array(test)
predictions_ar2=np.array(predictions_ar2)
#print(test[0:1])
plt.plot(test[0,:],'r-')
plt.plot(predictions_ar2[0,:],'g-')
plt.show()