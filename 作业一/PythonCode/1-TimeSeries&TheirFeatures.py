'''
威海近60天的天气情况（2020/8/6）
'''
import re
import requests
from matplotlib import pyplot as plt
import numpy as np
# 获取威海近60天的最低温和最高温
html = requests.get('https://www.yangshitianqi.com/weihai/60tian.html').text
#使用正则提取数据
pattern_temperature = r'<div class="fl i3 nz">(\d+~\d+)℃</div>'
pattern_date = r'<div class="t2 nz">(\d\d\.\d\d)</div>'
temperature = re.findall(pattern_temperature, html)
date = re.findall(pattern_date, html)
# 整理数据
max_d = [int(i.split('~')[1]) for i in temperature]
print(max_d)
min_d = [int(i.split('~')[0]) for i in temperature]
print(min_d)
# 定义图像质量
plt.figure(figsize=(9, 4.5), dpi=180)
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 绘制图像
plt.plot(max_d, 'r-')
plt.plot(min_d, 'b-')
# xy轴标识
plt.xlabel('date', size=24)
plt.ylabel('tem/℃', size=24)
plt.title('the latest 60 days in Weihai', size=24)
# 显示网格
plt.grid(axis='y')
# 显示图像
plt.show()

x = np.arange(1,60)
max_d = np.array(max_d)
min_d = np.array(min_d)
print(max_d)
print(min_d)
bar_width = 0.3
plt.bar(x, min_d, bar_width)
plt.bar(x+bar_width, max_d, bar_width, align="center")
plt.xlabel('date', size=24)
plt.ylabel('tem/℃', size=24)
plt.title('the latest 60 days in Weihai', size=24)
# 展示图片
plt.show()

