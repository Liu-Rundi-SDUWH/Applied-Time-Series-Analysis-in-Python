# '''
# Example 1: 趋势移除-以“洗发水数据集“为例
# '''
# from pandas import read_csv
# from pandas import datetime
# from matplotlib import pyplot
#
# # 1)加载数据
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
#
# series = read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(series)
# series.plot()
# pyplot.show()
#
# # 2）手动差分
# from pandas import read_csv
# from pandas import datetime
# from pandas import Series
# from matplotlib import pyplot
#
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return Series(diff)
#
# series = read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# X = series.values
# diff = difference(X)
# pyplot.plot(diff)
# pyplot.show()
#
# from pandas import read_csv
# from pandas import datetime
# from matplotlib import pyplot
#
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
#
# series = read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# diff = series.diff()
# pyplot.plot(diff)
# pyplot.show()
#
# '''
# Example 2: 单位根-以鲁商发展股票数据为例
# '''
# import pandas as pd
# import statsmodels.tsa.stattools as ts
# # 1)获取数据
# data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/鲁商发展.csv', index_col='Date')
#
# print(data.head())
# print(data.dtypes)
#
# # 2）原始数据单位根检验
# data = data['Close']
# result = ts.adfuller(data)
# print(result)
#
# '''
# Example 3:平滑法1
# '''
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x) + np.random.random(100) * 0.2
#
# from scipy.signal import savgol_filter
#
# yhat = savgol_filter(y, 51, 3)  # window size 51, polynomial order 3
#
# plt.plot(x, y)
# plt.plot(x, yhat, color='red')
# plt.show();
#
# def smooth(y, box_pts):
#     box = np.ones(box_pts) / box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
# plt.plot(x, y,'o', label='data')
# plt.plot(x, smooth(y, 3), 'r-', lw=2, label='smooth with box size 3')
# plt.plot(x, smooth(y, 19), 'g-', lw=2, label='smooth with box size 19')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.show();
#
# '''
# Example 4:平滑法2
# '''
# from astropy.modeling.models import Lorentz1D
# from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
# lorentz = Lorentz1D(1, 0, 1)
# x = np.linspace(-5, 5, 100)
#
# data_1D = lorentz(x) + 0.1 * (np.random.rand(100) - 0.5)
# gauss_kernel = Gaussian1DKernel(2)
# smoothed_data_gauss = convolve(data_1D, gauss_kernel)
#
# box_kernel = Box1DKernel(5)
# smoothed_data_box = convolve(data_1D, box_kernel)
#
# plt.plot(data_1D, label='Original')
# plt.plot(smoothed_data_gauss, label='Smoothed with Gaussian1DKernel')
# plt.plot(smoothed_data_box, label='Box1DKernel')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.show();
#
# smoothed = np.convolve(data_1D, box_kernel.array)
# plt.plot(data_1D, label='Original')
# plt.plot(smoothed, label='Smoothed with numpy')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.show();
#
#
#
# from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
# from astropy.modeling.models import Gaussian2D
#
# gauss = Gaussian2D(1, 0, 0, 3, 3)
# # Fake image data including noise
# x = np.arange(-100, 101)
# y = np.arange(-100, 101)
# x, y = np.meshgrid(x, y)
# data_2D = gauss(x, y) + 0.1 * (np.random.rand(201, 201) - 0.5)
#
# gauss_kernel = Gaussian2DKernel(2)
# smoothed_data_gauss = convolve(data_2D, gauss_kernel)
#
# tophat_kernel = Tophat2DKernel(5)
# smoothed_data_tophat = convolve(data_2D, tophat_kernel)
#
# import seaborn as sns
#
# pal = sns.light_palette("navy", as_cmap=True)
#
# ax = plt.axes()
# sns.heatmap(data_2D, ax=ax, cmap=pal)
# ax.set_title('Data')
# plt.show();
#
# ax = plt.axes()
# sns.heatmap(smoothed_data_gauss, ax=ax, cmap=pal)
# ax.set_title('Smoothed with Gaussian1DKernel')
# plt.show();
#
# ax = plt.axes()
# sns.heatmap(smoothed_data_tophat, ax=ax, cmap=pal)
# ax.set_title('Smoothed with Tophat2DKernel')
# plt.show();

import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn


#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,1400)

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

yy=fft(y)      #快速傅里叶变换
yreal = yy.real    # 获取实数部分
yimag = yy.imag    # 获取虚数部分

yf=abs(fft(y))    # 取绝对值
yf1=abs(fft(y))/len(x)   #归一化处理
yf2 = yf1[range(int(len(x)/2))] #由于对称性，只取一半区间

xf = np.arange(len(y))  # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))] #取一半区间


plt.subplot(221)
plt.plot(x[0:50],y[0:50])
plt.title('Original wave')

plt.subplot(222)
plt.plot(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B') #注意这里的颜色可以查询颜色代码表

plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')


plt.show()

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn



Fs = 150.0;     # sampling rate采样率
Ts = 1.0/Fs;    # sampling interval 采样区间
t = np.arange(0,1,Ts)  # time vector,这里Ts也是步长

ff = 25;     # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y)     # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T     # two sides frequency range
frq1 = frq[range(int(n/2))] # one side frequency range

YY = np.fft.fft(y)   # 未归一化
Y = np.fft.fft(y)/n   # fft computing and normalization 归一化
Y1 = Y[range(int(n/2))]

fig, ax = plt.subplots(4, 1)

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')

ax[1].plot(frq,abs(YY),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

ax[2].plot(frq,abs(Y),'G') # plotting the spectrum
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('|Y(freq)|')

ax[3].plot(frq1,abs(Y1),'B') # plotting the spectrum
ax[3].set_xlabel('Freq (Hz)')
ax[3].set_ylabel('|Y(freq)|')

plt.show()



