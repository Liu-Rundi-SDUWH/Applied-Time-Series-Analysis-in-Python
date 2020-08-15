'''
Call the required packages
'''
import numpy as np
import pandas as pd
import re
import requests
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import chi2

'''
Save your data
'''
import numpy as np
a=[1,2,3,4]
np.save('a.npy',a)
b=np.load('a.npy')
print('b is',b)

'''
Help when you do not know a command
'''
help(sum)

'''
Vectors, Matrix and other stuff
'''
# Create a vector
x=np.array([2,6,1,3,11])  #创建一个数组
print('x是',x)
print('x的形状是',x.shape)
y=x.T                   #数组的转置没有任何作用
print('y是',y)
print('x的形状是',y.shape)
x_t=x.reshape(1,5)       #通过reshape将其转化成数组
print('x_t是',x_t)
print('x_t的形状是',x_t.shape)

x1=x_t.T
print('x_t形状是',x1.shape)
#x1.T.shape	#返回数组形状

# Create a sequence
x2=np.arange(-4,5.1,1.5)  #与R语言的不同点：R语言包含5，py不包含5
print(x2)
#注：如果未指定步长：默认以1为间隔
x3=np.arange(-4,5.1)
print(x3)
#注：只有上界时：
x5=np.arange(6)      #从0到5，不含6
print(x5)
#注：如果想要序列具有特定的长度：(4, 5, length.out = 5)
x4=np.linspace(4,5,5)      #注：此处R与py都包含5这个上界
print(x4)

# Create duplicate numbers
#创建一个向量（一列数字）,该向量重复数字2 5次
x6=np.ones(5)
x6_1=x6*2
x6_2=x6_1.reshape(5,1)
print(x6)
print(x6_2.shape)
#创建一个向量（一列数字）,该向量重复数字3 4次
x7=np.ones(4)
x7_1=x7*3
x7_2=x7_1.reshape(4,1)
print(x7_2)
#下面的命令创建一个3 x 3的矩阵，在矩阵的每个元素上都是4 （4，3，3）
x8=np.ones((3,3))
x8_1=x8*4
print(x8_1)
#下面的命令将向量按照指定形状生成矩阵（创建一个2 x 2矩阵）。（11，5，9，2）
x9=np.array([11,5,9,2]).reshape(2,2)
print(x9)
#下面的命令使用向量元素创建向量矩阵。(4*1)
x10=np.array([11,5,9,2]).reshape(4,1)
print(x10)
#下面的命令使用向量元素创建向量矩阵。(1*4)
x11=np.array([11,5,9,2]).reshape(1,4)
print(x11)

# Create a specific sequence
#由0组成的数组
a=np.zeros((3,4))
#由1组成的数组
a1=np.ones((2,3,4),dtype=np.int16)	#3行4列2层
#向量拼接 (21，5，2，15）（- 3，- 6，1，- 7）(102，10，- 13，4） 4行3列
a2_1=np.array([21,5,2,15]).reshape(1,4)
a2_2=np.array([- 3,- 6,1,- 7]).reshape(1,4)
a2_3=np.array([102,10,- 13,4]).reshape(1,4)
a2=np.r_[a2_1,a2_2,a2_3]#按照行拼接row
print(a2)
a2_4=np.c_[a2_1.T,a2_2.T,a2_3.T]#按照列拼接column
print(a2_4)
#向量的第n个元素  从0开始
a3=np.array([21,5,2,15])
print(a3[3])
#向量第m到n个元素
print(a3[0:2])#不包含截止位置的元素
#向量特定位置的几个元素   两个位置的中括号
print(a3[[0,2,3]])
#m到n之间大小的元素
e=(a3>1) & (a3<16)
print(a3[e])
#矩阵第i行第j列元素
print(a2_1[0,2])#返回第一行第三列元素
#矩阵第i行
print(a2_1[0,:])#返回第一行第元素
#矩阵第j列
print(a2_1[:,2])#返回第三列元素

# Delete element & create an object containing a series of matrices
#假设您有一个向量，并且想要删除一些元素 (4，22，56，77，26，88，100）
b1=np.array([4,22,56,77,26,88,100])
#删除第4个    减掉1
b2=np.delete(b1,3)
b3=np.delete(b1,[3,4])
print(b2)
#删除第1-2个
b4=np.delete(b1,np.arange(0,1.1))
print(b4)
#创建l6个来自（0，1）正态分布的数，转成4*4矩阵
b5=np.random.normal(loc=0,scale=1,size=16)#均值mean,标准差std,数量
b5_1=b5.reshape(4,4)
print(b5_1)
#去掉2 4行
b5_2=np.delete(b5_1,[1,3],0)
print(b5_2)
#去掉2 4列
b5_3=np.delete(b5_1,[1,3],1)
print(b5_3)
#去掉1 4行，1 2列
b5_4=np.delete(b5_1,[0,3],0)
b5_5=np.delete(b5_4,[0,1],1)
print(b5_5)
#创建包含一系列矩阵的对象
#用1-8这8个数创建一个2*4的矩阵 1 ：3，Ç（2，4）
b6=np.arange(1,8.1)
b6_1=b6.reshape(2*4)
print(b6)
#用1-12这12个数创建3个2*2的矩阵对象 1 ：8，Ç（2，2，3）
b7=np.arange(1,12.1)
b7_1=b7.reshape(2,2,3)
print(b7_1)

'''
Matrix Computation
'''
#矩阵加，减，乘，除 （3，2，6） （- 2，- 1，5）
v1=np.array([3,2,6])
v2=np.array([-2,-1,5])
print(v1+v2)
print(v1-v2)
print(v1*v2)
print(v1/v2)
#可以使用符号*乘以具有不同尺寸的向量,创造矩阵（对应位置相乘）
v1_1=v1.reshape(1,3)
v2_1=v2.reshape(3,1)
print(v1_1*v2_1)
#当第一个矩阵的列数与第二个矩阵的行数相同时，作矩阵乘法(RNORM（6），2，3) (RNORM（9），3，3)
v3=np.random.normal(loc=0,scale=1,size=6)
v3_1=v3.reshape(2,3)
v4=np.random.normal(loc=0,scale=1,size=9)
v4_1=v4.reshape(3,3)
v5=np.matmul(v3_1,v4_1)
print(v5)
print(v5.shape)
#逆矩阵求法：  需要先转化成矩阵类型
v4_2=np.matrix(v4_1)
v6=v4_2.I
print(v6)

'''
Create a list of objects
'''
c1=3
c2=np.ones(15)
c3=np.arange(1,8.1)
c3_1=c3.reshape(2,4)
c3_2=np.matrix(c3_1)
c=[c1,c2,c3_2]
print(c)
print(c[0])

'''
Writing your own function
'''
#函数定义：这是一个简单的函数，返回x对象的均值
def my_mean(x):
    return sum(x)/len(x)
#函数调用
my_x=np.arange(1,4.1)
my_Mean=my_mean(my_x)
print(my_Mean)

'''
The for loop and the if condition in Python
'''
#for循环打印1：5
for i in range(1,6):
    print(i)
#runif（n，min，max）??在最小值和最大值之间生成n个随机均匀值。在下面的示例中，当数字<6时，您会看到打印件，否则看不到。
np.random.seed(12)
d1=np.random.uniform(3,4,(1,7))
print(d1)
index2=np.arange(0,d1.shape[1])
print(index2)
for i in index2:   #shape[0]表示行，shape[1]表示列
    if d1[0,i]<3.5 :
        print("小于3.5")
    elif (d1[0,i]>=3.5)&(d1[0,i]<3.7):
        print("大于3.5，小于3.7")
    else:
        print("大于等于3.7")

# Consider another example
#实例：
#下面创建随机均匀值的矢量y（20个元素）。其次，创建另一个称为z的向量（未知维），并且如果y的第i个值<= 0，则将相对的z[i]值指定为-1。否则，如果y> 0，则会将z [i]的值指定为1
np.random.seed(13)
y2=np.random.uniform(-10,10,(1,20))
print(y2)
z=[]
for i in range(0,20):
    if y2[0,i]<=0:
        z=np.append(z,-1)          #向量插入
    else:
        z=np.append(z,1)
print(z)