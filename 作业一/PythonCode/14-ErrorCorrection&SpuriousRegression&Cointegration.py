'''
Example 1:  Cointegrate-以“USB.csv"数据集为例
'''

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
data=pd.read_csv("./Book_dataset/USB.csv")

High = data["USB.High"]
Close = data["USB.Close"]

High_diff = np.diff(High)
Close_diff = np.diff(Close)
print(adfuller(High_diff))
print(adfuller(Close_diff))
print(coint(High,Close))


'''
Example 2: Estimatie-以“USB.csv"数据集为例
'''
import pandas as pd
import statsmodels.api as sm

data=pd.read_csv("./Book_dataset/USB.csv")
High = data["USB.High"]
Close = data["USB.Close"]

X = sm.add_constant(Close)
model1 = sm.OLS(High,X).fit()
print(model1.summary())
print(model1.params)

Y = sm.add_constant(High)
model2 = sm.OLS(Close,Y).fit()
print(model2.summary())
print(model2.params)


'''
Example 3: Error Correction Model-自行建立数据集
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(100)
x = np.random.normal(0, 1, 500)
y = np.random.normal(0, 1, 500)
X = pd.Series(np.cumsum(x)) + 100
Y = X + y + 30
for i in range(500):
    X[i] = X[i] - i/10
    Y[i] = Y[i] - i/10
plt.plot(X); plt.plot(Y);
plt.xlabel("Time"); plt.ylabel("Price");
plt.legend(["X", "Y"]);
plt.show()

plt.plot(Y-X);
plt.axhline((Y-X).mean(), color="red", linestyle="--");
plt.xlabel("Time"); plt.ylabel("Price");
plt.legend(["Y-X", "Mean"]);
plt.show()