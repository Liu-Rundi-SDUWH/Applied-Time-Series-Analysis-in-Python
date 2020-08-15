'''
Comparing forecasting performance
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize

data = pd.read_excel('CoronavirusSpreadUSAregions.xlsx')
y = data["NEWYORK"]
plt.xlabel('time')
plt.ylabel('num')
plt.title('New cases of Covid in NEWYORK')
y.plot(color='b',title='New cases of Covid in NEWYORK')
plt.show()

obs = len(y)-5
xxx = y[0:obs]
a = [i for i in range(obs)]
p = [i for i in range(obs)]
a[0] = xxx[0]
p[0] = 10000
k = [i for i in range(obs)]
v = [i for i in range(obs)]
v[0] = 0
n = len(xxx)
newseries = y

def myfunTheta0(x):
    z = w = 1
    likelihood = 0
    sigmae = 0
    for t in range(1, obs):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = xxx[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / len(newseries))
    return likelihood

def myfuTheta():
    v = lambda x: myfunTheta0(x)
    return v

x0 = np.asarray((0.6,0.2))
res = minimize(myfuTheta(), x0, method='SLSQP')

q = abs(res.x[0])
co = abs(res.x[1])
z = w = 1
sigmae = 0

for t in range(1,obs):
    k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
    p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
    v[t] = y[t] - z * a[t - 1]
    a[t] = co + w * a[t - 1] + k[t] * v[t]
    sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))

print("co",co)
print("the variance of e",sigmae/(len(newseries)-1))
print("the variance of u",q*(sigmae/(len(newseries)-1)))

MyForecasts = [0*i for i in range(5)]
MyForecasts[0] = a[obs-1]
MyForecasts[1] = co + MyForecasts[0]
MyForecasts[2] = co + MyForecasts[1]
MyForecasts[3] = co + MyForecasts[2]
MyForecasts[4] = co + MyForecasts[3]
plt.plot(np.array(y[len(y)-5:len(y)]),'k-')
plt.plot(MyForecasts,'r:')
plt.title("black is y_t, red is the forecasts")
plt.show()

Real = np.array(y[len(y)-5:len(y)])


def MASE(y,Real,Fore,m):
    h = len(Fore)
    fenzi,fenmu = 0,0
    for i in range(h):
        fenzi = fenzi + abs(Fore[i] - Real[i])
    fenzi = fenzi/h
    for i in range(m,n):
        fenmu = fenmu + abs(y[i] - y[i-m])
    fenmu = fenmu/(n-m)
    return fenzi/fenmu

def MAPE(Real,Fore):
    h = len(Fore)
    ratio = 0
    fenzi, fenmu = 0, 0
    for i in range(h):
        fenzi = abs(Fore[i]-Real[i])
        fenmu = abs(Fore[i]) + abs(Real[i])
        ratio = ratio + (fenzi/fenmu)
    return ratio*(200/h)

print(MASE(y,Real,MyForecasts,4))
print(MAPE(Real,MyForecasts))

import numpy.matlib
from scipy.optimize import minimize

a = np.matlib.zeros((obs,1))
a[0] = xxx[0]

def logLikConc0(x):
    w = z = 1
    co = 0
    v2 = 0
    for t in range(1,obs):
        v[t] = xxx[t] - z * a[t-1]
        v2 = v2 + v[t]**2
        a[t] = co + w*a[t-1] + x[0]*v[t]
    return (v2 - v[0])

def logLikConc():
    v = lambda x: logLikConc0(x)
    return v

def con(args):
    bmin, bmax = args
    cons = ({'type': 'ineq', 'fun': lambda b: b - bmin},{'type': 'ineq', 'fun': lambda b: -b + bmax})
    return cons

x0 = np.asarray((0.1))
args1 = (0,1)
cons = con(args1)
res = minimize(logLikConc(), x0, method='SLSQP',constraints=cons)

a = np.zeros(obs)
v = np.zeros(obs)


a[0] = xxx[0]
gamma = res.x[0]

for t in range(1,obs):
    v[t] = xxx[t] - z*a[t-1]
    a[t] = a[t-1] + gamma* v[t]

LLForecasts = [0*i for i in range(5)]
print(LLForecasts)

LLForecasts[0] = a[obs-1]
LLForecasts[1] = LLForecasts[0]
LLForecasts[2] = LLForecasts[1]
LLForecasts[3] = LLForecasts[2]
LLForecasts[4] = LLForecasts[3]

print('000',np.array(y[len(y)-5:len(y)]))
print('000',LLForecasts)
plt.plot(np.array(y[len(y)-5:len(y)]),'k-')
plt.plot(LLForecasts,'r:')
plt.title("black is y_t, red is the forecasts")
plt.show()

Real = np.array(y[len(y)-5:len(y)])
print(MASE(y,Real,LLForecasts,4))
print(MAPE(Real,LLForecasts))


'''
Forecast competion in action!
'''
import numpy as np
import math
import numpy.matlib
from scipy.optimize import minimize

def ForecastARkf0(x):
    a = [i for i in range(n)]
    p = [i for i in range(n)]
    a[0] = y[0]
    p[0] = 10000
    k = [i for i in range(n)]
    v = [i for i in range(n)]

    w = 1-math.exp(-abs(x[2]))

    likelihood = 0
    sigmae = 0
    z = 1
    for t in range(1, n):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = y[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def ForecastARkf1():
    v_ans = lambda x: ForecastARkf0(x)
    return v_ans


def ForcastARkf(y,h):
    n = len(y)
    x0 = np.asarray((0.2,1,2))
    res = minimize(ForecastARkf1(), x0, method='SLSQP')

    a = [i for i in range(n)]
    p = [i for i in range(n)]
    a[0] = y[0]
    p[0] = 10000
    k = [i for i in range(n)]
    v = [i for i in range(n)]
    v[0] = 0
    z = 1
    q = abs(res.x[0])
    co = abs(res.x[1])
    w = 1-math.exp(-abs(res.x[2]))
    sigmae = 0
    for t in range(1,n):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
        v[t] = y[t] - z * a[t - 1]
        a[t] = co + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))

    Forec = [0*i for i in range(h)]
    Forec[0] = a[len(y)]
    for i in range(1,h):
        Forec[i] = co + w*Forec[i-1]
    return Forec

##################################################
def logLikConc0(x):
    state = np.zeros(len(y))
    v = np.zeros(len(y))
    state[0] = y[0]

    w = 1-math.exp(-abs(x[2]))
    v2 = 0
    for t in range(1, len(y)):
        v[t] = y[t] - state[t-1]
        v2 = v2 + v[t]**2
        state[t] = x[1] + w*state[t-1] + x[2]* v[t]
    return (v2-v[0])

def logLikConc1():
    v_ans = lambda x: logLikConc0(x)
    return v_ans


def logLikConc(y,h):
    n = len(y)
    state = np.zeros(len(y))
    v = np.zeros(len(y))

    x0 = np.asarray((0.2,1,2))
    res = minimize(ForecastARkf1(), x0, method='SLSQP')

    w = 1-math.exp(-abs(res.x[0]))
    gamma = abs(res.x[1])
    co = abs(res.x[2])

    for t in range(1,len(y)):
        v[t] = y[t] - state[t - 1]
        state[t] = co + w * state[t - 1] + gamma * v[t]

    Forec = [0*i for i in range(h)]
    Forec[0] = state[len(y)]
    for i in range(1,h):
        Forec[i] = co + w*Forec[i-1]
    return Forec

######################################################
def logLikConc0(x):
    state = np.zeros(len(y))
    state[0] = y[0]
    v = np.zeros(len(y))
    w = 1
    v2 = 0
    for t in range(1, len(y)):
        v[t] = y[t] - state[t-1]
        v2 = v2 + v[t]**2
        state[t] = x[0] + w*state[t-1] + x[1]* v[t]
    return (v2-v[0])

def logLikConc1():
    v_ans = lambda x: logLikConc0(x)
    return v_ans


def ForecastTheta(y,h):
    n = len(y)
    state = np.zeros(len(y))
    v = np.zeros(len(y))

    x0 = np.asarray((0.3,1))
    res = minimize(ForecastARkf1(), x0, method='SLSQP')

    w = 1
    gamma = abs(res.x[0])
    co = abs(res.x[1])

    for t in range(1,len(y)):
        v[t] = y[t] - state[t - 1]
        state[t] = co + w * state[t - 1] + gamma * v[t]

    Forec = [0*i for i in range(h)]
    Forec[0] = state[len(y)]
    for i in range(1,h):
        Forec[i] = co + w*Forec[i-1]
    return Forec
#############################################################
def funcTheta0(x):
    k = np.zeros(len(y))
    p = np.zeros(len(y))
    a = np.zeros(len(y))
    v = np.zeros(len(y))
    a[0] = y[0]
    p[0] = 10000
    z = w = 1
    likelihood = 0
    for t in range(1, n):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = y[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def funcTheta1():
    v_ans = lambda x: logLikConc0(x)
    return v_ans


def ForecastThetakf(y,h):
    n = len(y)
    a = np.zeros(len(y))
    p = np.zeros(len(y))
    a[0] = y[0]
    p[0] = 10000
    k = np.zeros(len(y))
    v = np.zeros(len(y))
    v[0] = 0

    x0 = np.asarray((0.3,1))
    res = minimize(funcTheta1(), x0, method='SLSQP')

    z = w = 1
    q = abs(res.x[0])
    co = abs(res.x[1])

    for t in range(1,len(y)):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
        v[t] = y[t] - z * a[t - 1]
        a[t] = co + w * a[t - 1] + k[t] * v[t]


    Forec = [0*i for i in range(h)]
    Forec[0] = a[n]
    for i in range(1,h):
        Forec[i] = co + w*Forec[i-1]
    return Forec
#################################
def fmsoe0(x):
    obs = len(y)
    damped = np.matlib.zeros((obs, 2))
    damped[0, 0] = y[0]
    damped[0, 1] = 0
    inn = np.matlib.zeros((obs, 1))
    e2 = 0
    for t in range(1, obs):
        inn[t] = y[t] - damped[t-1,0] - x[2]* damped[t-1,1]
        damped[t,0] = damped[t-1,0] + x[2] * damped[t-1,1] + x[0] * inn[t]
        damped[t,1] = x[2]*damped[t - 1, 1] + x[1] * inn[t]
        e2 = e2 + inn[t]**2
    return (e2-inn[0]**2)/obs

def fmsoe1():
    v_ans = lambda x: logLikConc0(x)
    return v_ans


def ForecastDamped(y,h):
    obs = len(y)
    damped = np.matlib.zeros((obs,2))
    damped[0,0] = y[0]
    damped[0,1] = 0
    inn = np.matlib.zeros((obs, 1))

    x0 = np.asarray((np.random.rand(1),np.random.rand(1),np.random.rand(1)))
    res = minimize(fmsoe1(), x0, method='SLSQP')

    k1 = abs(res.x[0])
    k2 = abs(res.x[1])
    k3 = abs(res.x[2])

    if(k3>1): k3 = 1
    for t in range(1,obs):
        inn[t] = y[t] - damped[t - 1, 0] - k3 * damped[t - 1, 1]
        damped[t, 0] = damped + k3 * damped[t - 1, 1] + k1 * inn[t]
        damped[t, 1] = k3 * damped[t - 1, 1] + k2 * inn[t]

    Forec = [0*i for i in range(h)]
    Forec[0] = damped[obs-1,0] + k3* damped[obs-1,1]
    for i in range(1,h):
        Forec[i] = Forec[i-1] + damped[obs-1,1] * k3 **i
    return Forec