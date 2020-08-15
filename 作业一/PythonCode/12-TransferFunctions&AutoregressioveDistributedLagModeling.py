'''
Example 1: ARDL models-以“UK_Interest_Rates.xlsx"数据集为例
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_excel("./Book_dataset/UK_Interest_Rates.xlsx",usecols=[7,8,9])
target = pd.read_excel("./Book_dataset/UK_Interest_Rates.xlsx",usecols=[11])
print(data)
print(target)

X = data
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

'''
Example 2: ARDL models-以“global_temps.xlsx"数据集为例
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_excel("./Book_dataset/global_temps.xlsx",usecols=[1])
x = np.arange(1,len(data)+1)
print(data)
print(x)
x = x.reshape(-1, 1)

Y = data
X_train, X_test, y_train, y_test = train_test_split(x, data, test_size=0.7, random_state=1)
lr = LinearRegression()
lr.fit(x,data)
print(lr.coef_)
print(lr.intercept_)
