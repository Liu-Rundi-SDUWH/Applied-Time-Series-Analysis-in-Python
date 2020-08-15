'''
Seasonality
'''

'''
Additive seasonality
'''
import numpy as np
import matplotlib.pylab as plt
import math

np.random.seed(1213)
n = 102
e = np.sqrt(0.5)* np.random.randn(n)
u = np.sqrt(0.1)* np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
factor = [5, -4, 2,3]
s = 4
seasonal = (factor*(math.ceil(n/s)))[0:n]
y[0] = e[0] + seasonal[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = seasonal[t] + alpha[t-1] + e[t]
    alpha[t] = alpha[t-1] + u[t]
plt.plot(y,'k-')
plt.plot(alpha,'r-.')
plt.title("y and alpha")
plt.show()



import pandas as pd

y = np.asarray([6,2,1,3,7,3,2,4])
cma = np.zeros(len(y))
residuals = np.zeros(len(y))
print(residuals)
cma[2] = (0.5*y[0]+y[1]+y[2]+y[3]+0.5*y[4])/4
cma[3] = (0.5*y[1]+y[2]+y[3]+y[4]+0.5*y[5])/4
cma[4] = (0.5*y[2]+y[3]+y[4]+y[5]+0.5*y[6])/4
cma[5] = (0.5*y[3]+y[4]+y[5]+y[6]+0.5*y[7])/4
for i in range(2,6):
    residuals[i] = y[i] - cma[i]
index = [i for i in range(len(y))]
y = np.array(y)
cma = np.asarray(cma)
result = pd.DataFrame({'y':y,'cma':cma,'residuals':residuals},index= index)
print(result)



newseries = [0*i for i in range(len(y))]
factors = [residuals[4]-residuals[0],residuals[5]-residuals[1],residuals[2]-residuals[6],residuals[3]-residuals[7]]
factors = factors*2
for i in range(len(y)):
    newseries[i] = y[i] - factors[i]
plt.plot(y,'b-')
plt.plot(newseries,'r-.')
plt.title("cbind(y,newseries)")
plt.show()


cma = np.zeros(len(y))
for g in range(0,(len(y)-s)):
    sum = 0
    for h in range(g,g+s+1):
        sum = sum + w[h-g]* y[h]
    cma[int(g+s/2)] = sum

residuals = [0*i for i in range(len(cma))]
for i in range(2,len(cma)-2):
    residuals[i] = y[i] - cma[i]

factors = [0*i for i in range(s)]
index0 = np.arange(0,len(y),4)
index1 = np.arange(1,len(y),4)
index2 = np.arange(2,len(y),4)
index3 = np.arange(3,len(y),4)
sum0 = 0
index = np.arange(0,len(y),4),np.arange(1,len(y),4),np.arange(2,len(y),4),np.arange(3,len(y),4)
sum = [0*i for i in range(s)]
factors = [0*i for i in range(s)]

for seas in range(0,s):
    for i in index[seas]:
        sum[seas] = sum[seas] + residuals[i]

factors[0] = sum[0]/(len(index[0])-1)
factors[1] = sum[1]/(len(index[1])-2)
factors[2] = sum[2]/(len(index[2])-1)
factors[3] = sum[3]/(len(index[3]))

factors = (factors*math.ceil(n/s))[0:n]
newseries = [0*i for i in range(len(y))]
for i in range(len(y)):
    newseries[i] = y[i] - factors[i]
plt.plot(y,'b-')
plt.plot(newseries,'r-.')
plt.title("cbind(y,newseries)")
plt.show()



import numpy as np
import math
import matplotlib.pylab as plt

np.random.seed(243)
n = 87
e = np.sqrt(0.3)* np.random.randn(n)
u = np.sqrt(0.1)* np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
factor = [5, -4, 2,3]
s = 4
seasonal = (factor*(math.ceil(n/s)))[0:n]

y[0] = e[0] + seasonal[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = seasonal[t] + alpha[t-1] + e[t]
    alpha[t] = alpha[t-1] + u[t]

w = [1/(2*s)]
w = w*(s+1)
for i in range(1,s):
    w[i] = 1/s


cma = np.zeros(len(y))
for g in range(0,(len(y)-s)):
    sum = 0
    for h in range(g,g+s+1):
        sum = sum + w[h-g]* y[h]
    cma[int(g+s/2)] = sum

residuals = [0*i for i in range(len(cma))]
for i in range(2,len(cma)-2):
    residuals[i] = y[i] - cma[i]

factors = [0*i for i in range(s)]
index0 = np.arange(0,len(y),4)
index1 = np.arange(1,len(y),4)
index2 = np.arange(2,len(y),4)
index3 = np.arange(3,len(y),4)
sum0 = 0
index = np.arange(0,len(y),4),np.arange(1,len(y),4),np.arange(2,len(y),4),np.arange(3,len(y),4)
sum = [0*i for i in range(s)]
factors = [0*i for i in range(s)]

for seas in range(0,s):
    for i in index[seas]:
        sum[seas] = sum[seas] + residuals[i]

factors[0] = sum[0]/(len(index[0])-1)
factors[1] = sum[1]/(len(index[1])-2)
factors[2] = sum[2]/(len(index[2])-1)
factors[3] = sum[3]/(len(index[3]))

factors = (factors*math.ceil(n/s))[0:n]
newseries = [0*i for i in range(len(y))]
for i in range(len(y)):
    newseries[i] = y[i] - factors[i]
plt.plot(y,'b-')
plt.plot(newseries,'r-.')
plt.plot(alpha+e,'g:')
plt.title("y,newseries and alpha+e")
plt.show()

print(factor)
print(factors)



'''
Multiplicative seasonality
'''
import numpy as np
import math
import matplotlib.pylab as plt

np.random.seed(7)
n = 103
e = np.sqrt(0.5)* np.random.randn(n)
u = np.sqrt(0.4)* np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
factor = [1.7, 0.3, 1.9, 0.1]
seasonal = (factor*(int(100/4)))

y[0] = e[0]
alpha[0] = 5 + u[0]
for t in range(1,len(seasonal)):
    y[t] = seasonal[t] * (alpha[t-1] + e[t])
    alpha[t] = alpha[t-1] + u[t]
plt.plot(y,'m-')
plt.plot(alpha,'c:')
plt.title("y and alpha")
plt.show()


s = 4
n = len(y)
w = [1/(2*s)]
w = w*(s+1)
for i in range(1,s):
    w[i] = 1/s

cma = np.zeros(len(y))
for g in range(0,(len(y)-s)):
    sum = 0
    for h in range(g,g+s+1):
        sum = sum + w[h-g]* y[h]
    cma[int(g+s/2)] = sum

residuals = [0*i for i in range(len(cma))]
for i in range(2,len(cma)-2):
    residuals[i] = y[i] / cma[i]

factors = [0*i for i in range(s)]
index0 = np.arange(0,len(y),4)
index1 = np.arange(1,len(y),4)
index2 = np.arange(2,len(y),4)
index3 = np.arange(3,len(y),4)
sum0 = 0
index = np.arange(0,len(y),4),np.arange(1,len(y),4),np.arange(2,len(y),4),np.arange(3,len(y),4)
sum = [0*i for i in range(s)]
factors = [0*i for i in range(s)]

for seas in range(0,s):
    for i in index[seas]:
        sum[seas] = sum[seas] + residuals[i]

factors[0] = sum[0]/(len(index[0])-1)
factors[1] = sum[1]/(len(index[1])-2)
factors[2] = sum[2]/(len(index[2])-1)
factors[3] = sum[3]/(len(index[3]))
print('factor',factor)
print('factors',factors)
factors = (factors*math.ceil(n/s))[0:n]
newseries = [0*i for i in range(len(y))]
for i in range(len(y)):
    newseries[i] = y[i] / factors[i]

plt.plot(y,'b-')
plt.plot(newseries,'r-.')
plt.plot(alpha+e,'g:')
plt.title("y,newseries and alpha+e")
plt.show()
