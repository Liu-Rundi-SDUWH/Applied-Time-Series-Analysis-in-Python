'''
A quick introduction
'''
import matplotlib.pyplot as plt
import numpy as np

y = [3,4,1,5,2,10]
plt.plot(y,color = "b")
plt.xlabel('My x label')
plt.ylabel('My y label')
plt.title('My first plot :)')
plt.savefig("1-1.png")

n = 50
height = np.random.randint(30, size=n)
# print(height)
height1 = np.add(150, height)
height = height1.reshape(n, 1)
# print(height)

weight = np.random.randint(10, size=n)
# print(height)
weight = np.add(height1 / 3, weight)
weight = weight.reshape(n, 1)
# print(weight)


# 创建图形
plt.figure(1)

# 第一行第一列图形
ax1 = plt.subplot(3, 2, 1)
# 第一行第二列图形
ax2 = plt.subplot(3, 2, 2)
# 第二行
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)

# 选择ax1
plt.sca(ax1)
plt.plot(height, weight, 'o')

# 选择ax2
plt.sca(ax2)
plt.plot(height, weight, 'r+')
# plt.plot(height,weight,'+','k')


# 选择ax3
plt.sca(ax3)
plt.plot(height, weight, 'g*')

plt.sca(ax4)
plt.plot(height, weight, 'mD')

plt.sca(ax5)
plt.plot(y, 'g-')

plt.sca(ax6)
plt.plot(y, 'y--')

plt.show()
plt.savefig("1-2.png")


#设置图形宽度
bar_width = 0.7

n = 50
height = np.random.randint(30,size = n)
#print(height)
height = np.add(150,height)

weight = np.random.randint(10,size = n)
#print(height)
weight = np.add(height/3,weight)

#绘制图形
plt.bar(height,weight,bar_width,align='center',color='r')

# #加图例
plt.show()
plt.savefig("1-3.png")

values = [3,4,1,5,2,10]
# 包含每个柱子下标的序列
index = np.arange(6)
# 柱子的宽度
width = 0.45

p2 = plt.bar(index, values, width, color="#87CEFA")
plt.savefig("1-4.png")

# 均值与方差
mu, sigma = 100, 20
a = np.random.normal(mu, sigma, size=100)

# 绘制图形
plt.hist(a, 20, normed=0, histtype='bar', edgecolor='k', alpha=0.5)

plt.title('直方图')
plt.show()

# 直方图
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g')  # density=True 绘制并返回概率密度
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(40, 160)
plt.ylim(0, 0.03)
plt.grid(True)
plt.show()

box_1, box_2 = weight, height
plt.title('Examples of boxplot')  # 标题，并设定字号大小

# vert=False:水平箱线图；showmeans=True：显示均值
plt.boxplot([box_1, box_2], vert=False, showmeans=True)
plt.show()  # 显示图像

plt.boxplot([box_1, box_2], showmeans=True)
plt.show()  # 显示图像

elements = ["A", "B", "C", "D", "E"]
weight = [40, 15, 20, 10, 15]
colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#e6ab02"]

wedges, texts, autotexts = plt.pie(weight,
                                   autopct="%3.1f%%",
                                   textprops=dict(color="w"),
                                   colors=colors)

plt.setp(autotexts, size=15, weight="bold")
plt.setp(texts, size=12)

plt.title("FIGURE")
plt.show()

# 饼状图
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # 偏离相邻半径的大小

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


'''
Text analysis? A simple example to visualize words!
'''
import matplotlib.pyplot as plt
import numpy as np

width = 0.45
myspeech = "giacomo is my name yes giacomo and you call me giacomo since giacomo is my name"
newspeech = myspeech.split( )

count_set = set(newspeech)
count_list = list()
name_list = list()
for item in count_set:
    count_list.append((newspeech.count(item)))
    name_list.append(item)
print(count_list)
print(name_list)
plt.bar(name_list, count_list, width)
plt.show()


fig = plt.figure()
plt.pie(count_list,labels=name_list,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("Pie chart")

plt.show()
plt.savefig("PieChart.jpg")

colors = ['red', 'yellow', 'blue', 'green', 'gray']
colors.reverse()

plt.barh(name_list, count_list, tick_label=name_list, color=colors)
plt.show()



a1 = "If there is anyone out there who still doubts that America"
a2 = "is a place where all things are possible who still wonders"
a3 = "if the dream of our founders is alive in our time who still questions"
a4 = "the power of our democracy tonight is your answer It is the answer"
a5 = "told by lines that stretched around schools and churches in numbers"
a6 = "this nation has never seen by people who waited three hours and four"
a7 = "hours many for the first time in their lives because they believed that"
a8 = "this time must be different, that their voices could be that difference "

new_a1 = a1.split( )
new_a2 = a2.split( )
new_a3 = a3.split( )
new_a4 = a4.split( )
new_a5 = a5.split( )
new_a6 = a6.split( )
new_a7 = a7.split( )
new_a8 = a8.split( )

new = new_a1 + new_a2+ new_a3+ new_a4+ new_a5+ new_a6+ new_a7+ new_a8

count_set = set(new)
count_data = list()
count_list = list()
name_list = list()
for item in count_set:
    count_data.append([item,new.count(item)])

count_data.sort(key=lambda count_data : count_data[1])
# count_data = count_data[-10:]
print(count_data)
name_list = [count_data[i][0] for i in range(len(count_data))]
count_list = [count_data[i][1] for i in range(len(count_data))]
# print(name_list)
# print(count_list)
plt.figure(figsize=(8, 9))
plt.barh(name_list, count_list, tick_label=name_list)
plt.show()


a1 = "If there is anyone out there who still doubts that America"
a2 = "is a place where all things are possible who still wonders"
a3 = "if the dream of our founders is alive in our time who still questions"
a4 = "the power of our democracy tonight is your answer It is the answer"
a5 = "told by lines that stretched around schools and churches in numbers"
a6 = "this nation has never seen by people who waited three hours and four"
a7 = "hours many for the first time in their lives because they believed that"
a8 = "this time must be different, that their voices could be that difference "

new_a1 = a1.split( )
new_a2 = a2.split( )
new_a3 = a3.split( )
new_a4 = a4.split( )
new_a5 = a5.split( )
new_a6 = a6.split( )
new_a7 = a7.split( )
new_a8 = a8.split( )

new = new_a1 + new_a2+ new_a3+ new_a4+ new_a5+ new_a6+ new_a7+ new_a8
cut_list = ["is","by","I","a","if","If","in","the","that","of","our","be","there","this","their","and","they","your","hours","told","who"]

for i in cut_list:
    while i in new:
        new.remove(i)

count_set = set(new)
count_data = list()
count_list = list()
name_list = list()
for item in count_set:
    count_data.append([item,new.count(item)])

count_data.sort(key=lambda count_data : count_data[1])
count_data = count_data[-10:]
print(count_data)
name_list = [count_data[i][0] for i in range(len(count_data))]
count_list = [count_data[i][1] for i in range(len(count_data))]
# print(name_list)
# print(count_list)
plt.pie(count_list,labels=name_list,autopct='%1.2f%%')
plt.show()



import numpy as np
import pandas as pd
from scipy import stats
data = pd.read_excel('CoronavirusSpreadUSAregions.xlsx')
y = data["NEWYORK"]
plt.xlabel('time')
plt.ylabel('num')
plt.title('New cases of Covid in NEWYORK')
y.plot(color='r',title='New cases of Covid in NEWYORK')
plt.show()

'''
Exercise Human Resources data
'''
import numpy as np
edumat,jobmat, wag, age  = [],[],[],[]
n = 200
sexmat0 = np.random.rand(200)
sexmat = [round(i) for i in sexmat0]
sexmat = np.array(sexmat)
sexmat = sexmat.reshape(200,1)
print(sexmat)
for i in range(len(sexmat)):
    if(sexmat[i] == 0):
        edumat[i] = np.random.randint(2, 4)
        jobmat[i] = np.random.randint(1, 4)
        wag[i] = 2.3+0.5*np.random.rand(1)
        age[i] = round(40+10*np.random.rand(1))
    else:
        edumat[i] = np.random.randint(1, 4)
        jobmat[i] = np.random.randint(1, 2)
        wag[i] = 2+0.3*np.random.rand(1)
        age[i] = round(40+3.5*np.random.rand(1))



































