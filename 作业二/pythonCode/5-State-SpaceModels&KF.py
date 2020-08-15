'''
State-Space models and the Kalman filter
'''

'''
Kalman filter example
'''
import numpy as np
import matplotlib.pylab as plt
n = 100
np.random.seed(123)
e = np.sqrt(0.8)*np.random.randn(n)
u = np.sqrt(0.4)*np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = 0.9*alpha[t-1]+u[t]
plt.plot(y,'ro')
plt.plot(alpha,'b-')
plt.show()



n = 100
sigmae = 0.8
sigmau = 0.4
w = 0.9
z = 1

k = [0*i for i in range(n)]
v = [0*i for i in range(n)]
a = [0*i for i in range(n)]
p = [0*i for i in range(n)]
a[0] = 0
p[0] = 2.11
for t in range(1,n):
    k[t] = (z*w*p[t-1])/(z**2 *p[t-1]+sigmae)
    p[t] = w**2 * p[t-1] - w*z * k[t] * p[t-1] + sigmau
    v[t] = y[t] - z*a[t-1]
    a[t] = w*a[t-1] + k[t]*v[t]
plt.plot(y,'b-')
plt.plot(alpha,'g.')
plt.plot(a,'r-.')
plt.title("y,alpha,a")
plt.show()

'''
Concentrated Log-likelihood in action!
'''
import numpy as np
from scipy.optimize import minimize
import matplotlib.pylab as plt
n = 100
np.random.seed(61)
su = 0.1
se = 0.4
qreal = su/se
e = np.sqrt(se)*np.random.randn(n)
u = np.sqrt(su)*np.random.randn(n)
z = 1
wreal = 0.97
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = z*alpha[t-1] + e[t]
    alpha[t] = wreal*alpha[t-1]+u[t]
############  Standard Kalman filter approach  ##########################
a = [0*i for i in range(n)]
p = [0*i for i in range(n)]
a[0] = 0
p[0] = 10
k = [0*i for i in range(n)]
v = [0*i for i in range(n)]

def myfun0(x):
    z = 1
    likelihood = 0
    sigmae = 0
    for t in range(1, n):
        k[t] = (z * x[0] * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = x[0] ** 2 * p[t - 1] - x[0] * z * k[t] * p[t - 1] + x[1]
        v[t] = y[t] - z * a[t - 1]
        a[t] = x[0] * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t]**2/(z**2 * p[t-1] + 1))
        likelihood = likelihood + 0.5*np.log(2*np.pi)+0.5+0.5*np.log(z**2 * p[t-1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def myfu():
    v = lambda x: myfun0(x)
    return v

x0 = np.asarray((0.85,0.5))

res = minimize(myfu(), x0, method='SLSQP')
print(res)
print(res.x)


'''
State-Space models and the Kalman filter in action!
'''
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import pandas as pd
n = 100
np.random.seed(1265)
e = np.sqrt(0.1)*np.random.randn(n)
u = np.sqrt(0.05)*np.random.randn(n)
constant = 0.2
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = constant + 0.85*alpha[t-1] + u[t]
plt.plot(y,'ro')
plt.plot(alpha,'b-')
plt.show()

a = [0*i for i in range(n)]
p = [0*i for i in range(n)]
a[0] = 0
p[0] = 1
k = [0*i for i in range(n)]
v = [0*i for i in range(n)]



import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import pandas as pd
n = 100
np.random.seed(1265)
e = np.sqrt(0.1)*np.random.randn(n)
u = np.sqrt(0.05)*np.random.randn(n)
constant = 0.2
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = constant + 0.85*alpha[t-1] + u[t]
plt.plot(y,'ro')
plt.plot(alpha,'b-')
# plt.show()

a = [0*i for i in range(n)]
p = [0*i for i in range(n)]
a[0] = 0
p[0] = 1
k = [0*i for i in range(n)]
v = [0*i for i in range(n)]


def myfun0(x):
    z = 1
    likelihood = 0
    sigmae = 0
    for t in range(1, n):
        k[t] = (z * x[0] * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = x[0] ** 2 * p[t - 1] - x[0] * z * k[t] * p[t - 1] + x[1]
        v[t] = y[t] - z * a[t - 1]
        a[t] = x[2] + x[0] * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t]**2/(z**2 * p[t-1] + 1))
        likelihood = likelihood + 0.5*np.log(2*np.pi)+0.5+0.5*np.log(z**2 * p[t-1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def myfu():
    v = lambda x: myfun0(x)
    return v

x0 = np.asarray((0.9,1,0.1))
res = minimize(myfu(), x0, method='SLSQP')
# print(res)
# print(res.x)

v[1] = 0
w = abs(res.x[0])
q = abs(res.x[1])
co = abs(res.x[2])
from math import log,pi
z = 1
likelihood = sigmae = 0
for t in range(1,len(y)):
    k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
    p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
    v[t] = y[t] - z * a[t - 1]
    a[t] = co + w * a[t - 1] + k[t] * v[t]
    sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
    likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
sigmae = sigmae/len(y)
sigmau = q * sigmae
result = pd.DataFrame({'co':co,'w':w,'z':z,'sigmae':sigmae,'sigmau':sigmau},index=[0])
print(result)


'''
The Local level model (or simple exponential smoothing)
'''
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import pandas as pd
np.random.seed(153)
n = 100
e = np.sqrt(0.5)* np.random.randn(n)
u = np.sqrt(0.2)* np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = alpha[t-1] + u[t]
plt.plot(y,'g-')
plt.plot(alpha,'m-.')
plt.title("y and alpha")
plt.show()



import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import pandas as pd

a = [0*i for i in range(n)]
p = [0*i for i in range(n)]
a[0] = 0
p[0] = 10000
k = [0*i for i in range(n)]
v = [0*i for i in range(n)]


def myfun0(x):
    z = w = 1
    likelihood = 0
    sigmae = 0
    for t in range(1, n):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = y[t] - z * a[t - 1]
        a[t] = w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def myfu():
    v = lambda x: myfun0(x)
    return v

def con(args):
    qmin, qmax = args
    cons = ({'type': 'ineq', 'fun': lambda q: q - qmin},{'type': 'ineq', 'fun': lambda q: -q + qmax})
    return cons

x0 = np.asarray((0.2))
args = (0,1)
cons = con(args)
res = minimize(myfu(), x0, method='SLSQP',constraints=cons)
print(res)
print(res.x)



z = w = 1
likelihood = 0
sigmae = 0
q = res.x[0]
for t in range(1, n):
    k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
    p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
    v[t] = y[t] - z * a[t - 1]
    a[t] = w * a[t - 1] + k[t] * v[t]
    sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
print(sigmae/(n-1))
print(q*(sigmae)/(n-1))


'''
The Local Level with drift: (the Theta method)
'''
import numpy as np
import matplotlib.pylab as plt
n = 100
np.random.seed(572)
e = np.sqrt(0.8)*np.random.randn(n)
u = np.sqrt(0.1)*np.random.randn(n)
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
co = 0.1
y[0] = e[0]
alpha[0] = u[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = co + alpha[t-1]+u[t]
plt.plot(y,'c-')
plt.plot(alpha,'y-.')
plt.show()



a = [0*i for i in range(n)]
p = [0*i for i in range(n)]
a[0] = y[0]
p[0] = 10000
k = [0*i for i in range(n)]
v = [0*i for i in range(n)]
v[0] = 0

def myfunTheta0(x):
    z = w = 1
    likelihood = 0
    sigmae = 0
    for t in range(1, n):
        print(t)
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = y[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def myfuTheta():
    v = lambda x: myfunTheta0(x)
    return v

x0 = np.asarray((0.6,0.2))
res = minimize(myfuTheta(), x0)


q = abs(res.x[0])
co = abs(res.x[1])
z = w = 1
sigmae = 0
for t in range(1,len(y)):
    k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
    p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
    v[t] = y[t] - z * a[t - 1]
    a[t] = co + w * a[t - 1] + k[t] * v[t]
    sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))

print(sigmae/(n-1))
print(q*(sigmae/(n-1)))



def generateTheta(n,sigmae,sigmau,co):
    e = np.sqrt(sigmae)*np.random.randn(n)
    u = np.sqrt(sigmau)*np.random.randn(n)
    y = [0*i for i in range(n)]
    alpha = [0*i for i in range(n)]
    y[0] = e[0]
    alpha[0] = u[0]
    for t in range(1,n):
        alpha[t] = co + alpha[t-1] + u[t]
        y[t] = alpha[t-1] + e[t]
    return y




import numpy as np
from scipy.optimize import minimize
import pandas as pd

def myfunTheta0(x):
    n = len(y)
    z = w = 1
    likelihood = 0
    sigmae = 0

    a = [0*i for i in range(n)]
    p = [0*i for i in range(n)]
    a[0] = 0
    p[0] = 0
    k = [0 * i for i in range(n)]
    v = [0 * i for i in range(n)]

    for t in range(1, n):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + x[0]
        v[t] = y[t] - z * a[t - 1]
        a[t] = x[1] + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
        likelihood = likelihood + 0.5 * np.log(2 * np.pi) + 0.5 + 0.5 * np.log(z ** 2 * p[t - 1] + 1)
    likelihood = likelihood + 0.5 * n * np.log(sigmae / n)
    return likelihood

def myfuTheta():
    v = lambda x: myfunTheta0(x)
    return v

def EstimateTheta(y):
    n = len(y)
    x0 = np.asarray((0.5, 0.2))
    res = minimize(myfuTheta(), x0)
    v = [0 * i for i in range(n)]
    v[0] = 0
    w = z = 1
    q = abs(res.x[0])
    co = abs(res.x[1])
    sigmae = 0

    a = [0*i for i in range(n)]
    p = [0*i for i in range(n)]
    a[0] = 0
    p[0] = 0
    k = [0 * i for i in range(n)]
    v = [0 * i for i in range(n)]

    for t in range(1,len(y)):
        k[t] = (z * w * p[t - 1]) / (z ** 2 * p[t - 1] + 1)
        p[t] = w ** 2 * p[t - 1] - w * z * k[t] * p[t - 1] + q
        v[t] = y[t] - z * a[t - 1]
        a[t] = co + w * a[t - 1] + k[t] * v[t]
        sigmae = sigmae + (v[t] ** 2 / (z ** 2 * p[t - 1] + 1))
    sigmae = sigmae/len(y)
    sigmau = q*sigmae
    thelist = pd.DataFrame({'sigmae':sigmae,'sigmau':sigmau,'co':co},index=[0])
    return  thelist

np.random.seed(11)
y = generateTheta(100,.6,.2,1)
ans = EstimateTheta(generateTheta(100,.6,.2,1))
print(ans)


'''
The exponential smoothing with one source of error
'''
import numpy as np
import matplotlib.pylab as plt
np.random.seed(213)
n = 100
e = np.sqrt(0.6)*np.random.randn(n)
gamma = 0.3
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = e[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = alpha[t-1] + gamma * e[t]
plt.plot(y,'b-')
plt.plot(alpha,'r-.')
plt.title("y and alpha")
plt.show()



import numpy.matlib
from scipy.optimize import minimize

a = [0*i for i in range(n)]
a[0] = y[0]
e = np.matlib.zeros((len(y),1))

def myfun0(x):
    e2 = 0
    for t in range(1, n):
        e[t] = y[t] - a[t-1]
        e2 = e2+ (y[t] - a[t-1])**2
        a[t] = a[t-1] + x[0] *e[t]
    return e2/n

def myfu():
    v = lambda x: myfun0(x)
    return v

def con(args):
    bmin, bmax = args
    cons = ({'type': 'ineq', 'fun': lambda b: b - bmin},{'type': 'ineq', 'fun': lambda b: -b + bmax})
    return cons

x0 = np.asarray((0.2))
args1 = (0,1)
cons = con(args1)
res = minimize(myfu(), x0, method='SLSQP',constraints=cons)
print(res)
print(res.x)


'''
The Theta method with one source of error
'''
import numpy as np
import matplotlib.pylab as plt

np.random.seed(5)
n = 100
e = np.sqrt(0.4)* np.random.randn(n)
gamma = 0.1
con = 0.05
y = [0*i for i in range(n)]
alpha = [0*i for i in range(n)]
y[0] = e[0]
alpha[0] = gamma*e[0]
for t in range(1,n):
    y[t] = alpha[t-1] + e[t]
    alpha[t] = con + alpha[t-1] + gamma*e[t]
plt.plot(y,'b-')
plt.plot(alpha,'r-.')
plt.title("y and alpha")
plt.show()



import numpy.matlib
from scipy.optimize import minimize
a = [0*i for i in range(n)]
a[0] = y[0]
e = np.matlib.zeros((len(y),1))

def myfun0(x):
    e2 = 0
    for t in range(1, n):
        e[t] = y[t] - a[t-1]
        e2 = e2 + (y[t] - a[t-1])**2
        a[t] = x[1] + a[t-1] + x[0] *e[t]
    return e2/n

def myfu():
    v = lambda x: myfun0(x)
    return v

x0 = np.asarray((0.2,0.1))
res = minimize(myfu(), x0, method='SLSQP')
print(res)
print(res.fun)
print(res.x)