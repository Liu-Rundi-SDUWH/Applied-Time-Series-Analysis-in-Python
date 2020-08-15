'''
Example 1: Forecast-以“obesity.xlsx"为例
'''
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("./Book_dataset/obesity.xlsx")
x1 = data.x1
x2 = data.x2
x3 = data.x3
y1 = []
y2 = []
t = []

for i in range(len(x1)):
    y1.append(math.log(x1[i]/x3[i]))
    y2.append(math.log(x2[i]/x3[i]))
    t.append(i+1)

plt.plot(t,x1,label="BMI <25")
plt.plot(t,x2,label="BMI 25~30")
plt.plot(t,x3,label="BMI >30")


f1 = np.polyfit(t, y1, 3)
print('f1 is :\n',f1)

f2 = np.polyfit(t, y2, 3)
print('f2 is :\n',f2)

predict_t = [23,24,25,26,27,28,29]
predict_y1 = []
predict_y2 = []
for i in predict_t:
    predict_y1.append(np.polyval(f1,i))
    predict_y2.append(np.polyval(f2,i))
x1_x3 = []
x2_x3 = []
pre_x3 = []
pre_x1 = []
pre_x2 = []
for i in range(7):
    x1_x3.append(math.e**predict_y1[i])
    x2_x3.append(math.e**predict_y2[i])
    pre_x3.append(1/(1+x1_x3[i]+x2_x3[i])*100)
    pre_x1.append(pre_x3[i]*x1_x3[i])
    pre_x2.append(pre_x3[i]*x2_x3[i])

print(pre_x1)
plt.plot(predict_t,pre_x1,'--',label="BMI <25")
plt.plot(predict_t,pre_x2,'--',label="BMI 25~30")
plt.plot(predict_t,pre_x3,'--',label="BMI >30")
plt.legend()
plt.show()

'''
Example 2: Model-以“UK_expand.xlsx"数据为例
'''
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_excel("./Book_dataset/UK_expend.xlsx")
data2 = pd.read_excel("./Book_dataset/UK_expend.xlsx",usecols=[5,6,7])
c = data.cons
i = data.inv
g = data.gov
x = data.other
y1 = []
y2 = []
y3 = []
t = []

for l in range(len(c)):
    t.append(l)
    y1.append(math.log(c[l]/x[l]))
    y2.append(math.log(i[l]/x[l]))
    y3.append(math.log(g[l]/x[l]))

plt.plot(y1,label="y1")
plt.plot(y2,label="y2")
plt.plot(y3,label="y3")
plt.legend()
plt.show()

dy1 = data.dy1
dy2 = data.dy2
dy3 = data.dy3
print(data)
orgMod = sm.tsa.VARMAX(data2,order=(3,0),exog=None)
fitMod = orgMod.fit()
print(fitMod.summary())

# 获得模型残差
resid = fitMod.resid
result = {'fitMod':fitMod,'resid':resid}
print(result)

'''
Example 3: Model-以“nao.xlsx"数据为例
'''
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("./Book_dataset/nao.xlsx")
x1 = data['data']
y1 = []
t = []

for i in range(len(x1)):
    y1.append(x1[i])
    t.append(i+1)

plt.plot(t[0:700],x1[0:700])
f1 = np.polyfit(t[0:700], y1[0:700], 3)
print('f1 is :\n',f1)

predict_t = np.arange(700,700+len(y1[700:]))
predict_y1 = []
for i in predict_t:
    predict_y1.append(np.polyval(f1,i))

print(predict_y1)
plt.plot(predict_t,predict_y1,'--')
plt.legend()
plt.show()