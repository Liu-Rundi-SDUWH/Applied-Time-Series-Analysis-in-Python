'''
Seasonal-state-Space representation
'''
import numpy as np
import matplotlib.pylab as plt

np.random.seed(55)
n = 100
e = np.sqrt(.4) * np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
s = 4
sfactor = np.array(10 * np.random.rand(s))
y[0] = sfactor[0] + e[0]
y[1] = sfactor[1] + e[1]
y[2] = sfactor[2] + e[2]
y[3] = sfactor[3] + e[3]
alpha[0] = sfactor[0] + 0.2 * e[0]
alpha[1] = sfactor[1] + 0.2 * e[1]
alpha[2] = sfactor[2] + 0.2 * e[2]
alpha[3] = sfactor[3] + 0.2 * e[3]

for t in range(4,n):
    alpha[t] = alpha[t - s] + 0.3 * e[t]
    y[t] = alpha[t-s]+e[t]

plt.plot(y[0:n],'k-')
plt.plot(alpha[0:n],'r:')
plt.xlabel("Index")
plt.ylabel("y[1:n]")
plt.show()



import numpy.matlib
from scipy.optimize import minimize

s = 4
state = np.matlib.zeros((n,1))
e = np.matlib.zeros((n,1))

a = [0*i for i in range(n)]
a[0] = y[0]


def logLikConc0(x):
    e2 = 0
    for t in range(s, n):
        e[t] = y[t] - state[t-s]
        e2 = e2 + e[t]**2
        state[t] = state[t-s] + x[0] * e[t]
    return (e2-e[0])/(n-1)

def logLikConc():
    v = lambda x: logLikConc0(x)
    return v

def con(args):
    bmin, bmax = args
    cons = ({'type': 'ineq', 'fun': lambda b: b - bmin},{'type': 'ineq', 'fun': lambda b: -b + bmax})
    return cons

x0 = np.asarray((0.4))
args1 = (0,1)
cons = con(args1)
res = minimize(logLikConc(), x0, method='SLSQP',constraints=cons)
print(res)
print("this is the estimated gamma",res.x)
print("this is the estimated variance of e",res.fun)



import numpy as np
import matplotlib.pylab as plt

np.random.seed(1132)
n = 100
e = np.sqrt(.4) * np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
s = 4
co = 0.3
sfactor = np.array(10 * np.random.rand(s))
y[0] = sfactor[0] + e[0]
y[1] = sfactor[1] + e[1]
y[2] = sfactor[2] + e[2]
y[3] = sfactor[3] + e[3]
alpha[0] = sfactor[0] + 0.2 * e[0]
alpha[1] = sfactor[1] + 0.2 * e[1]
alpha[2] = sfactor[2] + 0.2 * e[2]
alpha[3] = sfactor[3] + 0.2 * e[3]

for t in range(4,n):
    alpha[t] = co + alpha[t - s] + 0.1 * e[t]
    y[t] = alpha[t-s]+e[t]

plt.plot(y[0:n],'g-')
plt.plot(alpha[0:n],'m:')
plt.xlabel("Index")
plt.ylabel("y[1:n]")
plt.show()




import numpy.matlib
from scipy.optimize import minimize

s = 4
v = np.matlib.zeros((n,1))
state = np.matlib.zeros((n,1))
for i in range(s):
    state[i] = y[i]

def logLikConc0(x):
    e2 = 0
    for t in range(s, n):
        v[t] = y[t] - state[t-s]
        e2 = e2 + e[t]**2
        state[t] = x[1] + state[t-s] + x[0] * v[t]
    return (e2-e[0])/(n-1)

def logLikConc():
    v = lambda x: logLikConc0(x)
    return v

x0 = np.asarray((0.2,0.2))
res = minimize(logLikConc(), x0, method='SLSQP')
print(res)


'''
Forecasting time series
'''
import numpy as np
import matplotlib.pylab as plt

np.random.seed(1347)
n = 105
e = np.sqrt(.5) * np.random.randn(n)
u = np.sqrt(.1) * np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
co = 0.06
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    alpha[t] = co + alpha[t-1] + u[t]
    y[t] = alpha[t-1] + e[t]
plt.plot(y,'b-')
plt.show()



import numpy as np
import numpy.matlib
from scipy.optimize import minimize

obs = 100
xxx = [0*i for i in range(obs)]
for i in range(obs):
    xxx[i] = y[i]
print(len(xxx))
a = [0*i for i in range(obs)]
p = [0*i for i in range(obs)]
a[0] = xxx[0]
p[0] = 10000
k = [0*i for i in range(obs)]
v = [0*i for i in range(obs)]

def funcTheta0(x):
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1,obs):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = xxx[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def funcTheta():
    v = lambda x: funcTheta0(x)
    return v

x0 = np.asarray((0.6,0.2))
res = minimize(funcTheta(), x0, method='SLSQP')
print(res)
print(res.x)

q = abs(res.x[0])
co = abs(res.x[1])
z = w = 1
sigmae = 0

for t in range(1,obs):
    k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
    p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
    v[t] = y[t] - z * a[t - 1]
    a[t] = co + w * a[t - 1] + k[t] * v[t]
    sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t] + 1))
print("co",co)
print("the variance of e",sigmae/(obs-1))
print("the variance of u",q*(sigmae/(obs-1)))

MyForecasts = [0*i for i in range(5)]
MyForecasts[0] = a[obs-1]
MyForecasts[1] = co + MyForecasts[0]
MyForecasts[2] = co + MyForecasts[1]
MyForecasts[3] = co + MyForecasts[2]
MyForecasts[4] = co + MyForecasts[3]
plt.plot(y[100:105],'b-')
plt.plot(MyForecasts,'r:')
plt.show()


'''
Forecasting seasonal series
'''
import numpy as np
import matplotlib.pylab as plt
import math

np.random.seed(115)
n = 106
e = np.sqrt(0.4)* np.random.randn(n)
u = np.sqrt(0.1)* np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
con = 0.2
factor = [0.7, 2, 0.2 ,1.1]
seasonal = (factor*(math.ceil(n/4)))[0:n]
y[0] = e[0] + seasonal[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = seasonal[t] *(alpha[t-1] + e[t])
    alpha[t] = con + alpha[t-1] + u[t]
plt.plot(y,'k-')
plt.plot(alpha+e,'r-.')
plt.title("y and alpha+e")
plt.show()

x = y[0:100]
s = 4
n = len(x)
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
# print('000',len(y))
# print('000',len(factors))
for i in range(len(factors)):
    newseries[i] = y[i] / factors[i]

a = [0*i for i in range(len(newseries))]
p = [0*i for i in range(len(newseries))]
a[0] = newseries[0]
p[0] = 10000
k = [0*i for i in range(len(newseries))]
v = [0*i for i in range(len(newseries))]
v[0] = 0

from scipy.optimize import minimize

def funcTheta0(x):
    z = w = 1
    likelihood = sigmae = 0
    for t in range(1,len(newseries)):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = newseries[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / len(newseries))
    return likelihood

def funcTheta():
    v = lambda x: funcTheta0(x)
    return v

x0 = np.asarray((0.6,0.2))
res = minimize(funcTheta(), x0, method='SLSQP')
print(res)
print(res.x)

q = abs(res.x[0])
co = abs(res.x[1])
z = w = 1
sigmae = 0

for t in range(1,len(newseries)):
    k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
    p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
    v[t] = y[t] - z * a[t - 1]
    a[t] = co + w * a[t - 1] + k[t] * v[t]
    sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t] + 1))
print("co",co)
print("the variance of e",sigmae/(len(newseries)-1))
print("the variance of u",q*(sigmae/(len(newseries)-1)))



w = z = 1
Myforecasts = [0*i for i in range(6)]
Myforecasts[0] = a[len(x)]
for o in range(1,6):
    Myforecasts[o] = co + Myforecasts[o-1]
print(np.array(Myforecasts))
print(factors)
SeasonalForcast = np.array(Myforecasts)* np.array(factors[0:6])
print(SeasonalForcast)
plt.plot(y[100:106],'k-')
plt.plot(SeasonalForcast,'r-.')
plt.title("black is y_t, red is the forecasts")
plt.show()


