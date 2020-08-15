'''
The Normal distribution
'''
import numpy as np
import math
import matplotlib.pylab as plt

x = np.arange(-3.8,4.2,1/100)
mu = 0.2
sigma = np.sqrt(1.5)

pdf, cdf = list(), list()
for i in range(len(x)):
    pdf.append(1/np.sqrt(2*math.pi*sigma**2)*math.exp(-0.5*(x[i]-mu)**2/sigma**2))
    cdf.append(round(np.sum(pdf)/100,4))
plt.plot(pdf,'r-')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Probability distribution function")
plt.show()

plt.plot(cdf,'b-')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cumulative distribution function")
plt.show()

'''
Bivariate Normal distribution
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
mx = 0
my = 0
varx = 0.5
vary = 0.6
covxy = -0.3
Sigma = np.matlib.ones((2,2))
Sigma[0,0] = varx
Sigma[1,1] = vary
Sigma[0,1] = Sigma[1,0] = covxy

x = np.arange(-4,4,8/100)
y = np.arange(-4,4,8/100)

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')


X, Y = np.meshgrid(x, y)
Z = (1/(2*np.pi*np.linalg.det(Sigma)**0.5))*np.exp(-0.5*((vary*(X-mx)**2+(Y-my)*(-2*(X-mx)*covxy+varx*(Y-my)))/(varx*vary-2*covxy)))

ax3.plot_surface(x,y,Z,cmap='rainbow')
plt.show()



from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

#作图
ax4.plot_surface(X,Y,Z,alpha=0.3,cmap='winter')     #生成表面， alpha 用于控制透明度
ax4.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面

#设定显示范围
ax4.set_xlabel('X')
ax4.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax4.set_ylabel('Y')
ax4.set_ylim(-4, 6)
ax4.set_zlabel('Z')
ax4.set_zlim(-3, 3)

plt.show()
