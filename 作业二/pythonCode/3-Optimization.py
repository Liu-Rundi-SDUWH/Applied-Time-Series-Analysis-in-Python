'''
Optimization with Python
'''
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize

np.random.seed(1)
n = 100
x = 1+7*np.random.rand(n)
y = 1*np.random.randn(n) + 0.8 * x

plt.plot(x,y,'o')
plt.show()


from scipy.optimize import minimize

def myfu(args):
    x , y = args
    # v = (y - b*x)*(y - b*x)
    v = lambda b: np.sum((y - b*x)*(y - b*x))
    return v

args = (x,y)
x0 = np.asarray((1))  # 初始猜测值
res = minimize(myfu(args), x0, method='SLSQP')
print(res.fun)
print(res.success)
print(res.x)


def myfu(args):
    x , y = args
    # v = (y - b*x)*(y - b*x)
    v = lambda b: np.sum((y - b*x)*(y - b*x))
    # v = lambda b: np.sum(abs(y - b * x))
    return v

args = (x,y)
x0 = np.asarray((1))  # 初始猜测值
res = minimize(myfu(args), x0, method='SLSQP')
print(res.fun)
print(res.success)
print(res.x)


'''
Exercise (unidimensional)
'''
import numpy as np
import matplotlib.pylab as plt

np.random.seed(1224)
x = [i/100 for i in range(1,101)]
x = np.sin(np.array(x))
b = np.random.rand(1)
y = -b *x + 0.03*(np.random.randn(len(x)))

plt.plot(x,y,'ro')
plt.show()


def myfu(args):
    x , y = args
    v = lambda b: -(np.sum((y - b*x)*(y - b*x)))
    return v

def con(args):
    bmin, bmax = args
    cons = ({'type': 'ineq', 'fun': lambda b: b - bmin},{'type': 'ineq', 'fun': lambda b: -b + bmax})
    return cons

args = (x,y)
x0 = np.asarray((1))  # 初始猜测值
args1 = (-14,14)
cons = con(args1)
res = minimize(myfu(args), x0, method='SLSQP',constraints=cons)
print(res.fun)
print(res.success)
print(res.x)

np.random.seed(1224)
x = [i/100 for i in range(1,101)]
x = np.sin(np.array(x))
b = np.random.rand(1)
y = -b *x + 0.03*(np.random.randn(len(x)))

plt.plot(x,y,'ro')
# plt.show()

def myfu(args):
    x , y = args
    v = lambda b: -(np.sum((y - b*x)*(y - b*x)))
    return v

def con(args):
    bmin, bmax = args
    cons = ({'type': 'ineq', 'fun': lambda b: b - bmin},{'type': 'ineq', 'fun': lambda b: -b + bmax})
    return cons

args = (x,y)
x0 = np.asarray((1))  # 初始猜测值
args1 = (-14,14)
cons = con(args1)
res = minimize(myfu(args), x0, method='SLSQP',constraints=cons)
print(res.fun)
print(res.success)
print(res.x)



def fun(args):
    a, b = args
    v = lambda x: (a + x[0])**2 + (b + x[1])**2
    return v

# 定义常量值
args = (-2,4)  # a,b,c,d
# 设置初始猜测值
x0 = np.asarray((1,3))

res = minimize(fun(args), x0, method='SLSQP')
print(res.fun)
print(res.success)
print(res.x)



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,10,15/50)
yy = np.arange(-11,2,13/50)
X, Y = np.meshgrid(xx, yy)
# print(X)
Z = (-2+X)**2 + (4+Y)**2

#作图
ax3.plot_surface(xx,yy,Z,cmap='rainbow')
# ax3.contour(xx,yy,Z, offset=-2,cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
plt.show()



from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

#生成三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图
ax4.plot_surface(X,Y,Z,alpha=0.3,cmap='winter')     #生成表面， alpha 用于控制透明度
ax4.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面
ax4.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
ax4.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
#ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数

#设定显示范围
ax4.set_xlabel('X')
ax4.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax4.set_ylabel('Y')
ax4.set_ylim(-4, 6)
ax4.set_zlabel('Z')
ax4.set_zlim(-3, 3)

plt.show()



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

n_radii = 8
n_angles = 36

radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

x = np.append(0, (radii * np.cos(angles)).flatten())
y = np.append(0, (radii * np.sin(angles)).flatten())
z = np.sin(-x * y)
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True,cmap="rainbow")
plt.show()
