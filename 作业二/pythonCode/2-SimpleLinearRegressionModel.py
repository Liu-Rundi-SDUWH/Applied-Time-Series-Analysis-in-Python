'''
Simple linear regression model
'''
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
pointsline = list()
x = stats.chi2.rvs(6,size=40)
e = stats.norm.rvs(size=40)

y = 1.3+0.8*x+e
pointsline = [1.3+0.8*i for i in range(21)]
# print(list(y))
y = list(y)
print(y)
plt.plot(x,y,'o')
plt.plot(pointsline)
plt.show()



n = 200
sex = list()
appreciation = list()
data0 = list()
data1 = list()
data0_x = list()
data1_x = list()
temperature = [0,]
gender = np.random.rand(n)
gender = np.round(1.1*gender)

for i in range(len(gender)):
    if gender[i] == 0:
        sex.append("Femall")
    else:
        sex.append(("Male"))

temperature = [2+(x*5)/100 for x in range(0,200)]
ef = 0.4* stats.norm.rvs(size=n)
em = 0.8* stats.norm.rvs(size=n)

ef = list(ef)
em = list(em)

i = 0
for i in range(n):
    if(gender[i] == 0):
        appreciation.append(7 - 0.3 * temperature[i] + ef[i])
    else:
        appreciation.append(10 - 0.7 * temperature[i] + em[i])
plt.plot(temperature,appreciation,'o')
plt.xlabel("temperature")
plt.ylabel("appreation")
plt.show()

for i in range(len(sex)):
    if(gender[i] == 0):
        data0.append(appreciation[i])
        data0_x.append(temperature[i])
    else:
        data1.append(appreciation[i])
        data1_x.append(temperature[i])

ax = plt.subplot()

ax.plot(data0_x,data0, 'o',label='Female')
ax.plot(data1_x,data1, 'ro',label='Male')

plt.xlabel("temperature")
plt.ylabel("appreation")
plt.legend()
plt.show()



import pandas as pd
OurRegression = np.polyfit(appreciation,temperature,1)
p1 = np.poly1d(OurRegression)
print(OurRegression)
print(p1)



summary_across_rows = pd.DataFrame(OurRegression).describe() # across axis=0
print(summary_across_rows)


OurRegression = np.polyfit(appreciation,temperature,1)
p1 = np.poly1d(OurRegression)
print(OurRegression)
print(p1)

resid = list()
for i in range(len(appreciation)):
    resid.append(temperature[i] - p1(appreciation[i]))
print(resid)

# 绘制图形
plt.hist(resid, 30, edgecolor='blue',facecolor = 'pink', alpha=0.5)

plt.title('histogram of residuals')
plt.show()



import pandas as pd
n = 200
sex = list()
appreciation0 = list()
appreciation1 = list()
temperature0 = list()
age0 = list()
total = list()


gender = np.random.rand(n)
gender = np.round(1.1*gender)

for i in range(len(gender)):
    if gender[i] == 0:
        sex.append("Femall")
    else:
        sex.append(("Male"))

temperature = [2+(x*5)/100 for x in range(0,200)]
age = 22 + stats.chi2.rvs(5,size=n)
ef = 0.4* stats.norm.rvs(size=n)
em = 0.8* stats.norm.rvs(size=n)

ef = list(ef)
em = list(em)

for i in range(n):
    if(gender[i] == 0):
        appreciation0.append(7 - 0.3 * temperature[i] + 0.1*age[i]+ef[i])
        temperature0.append(temperature[i])
        age0.append(age[i])
        total.append(temperature[i]+age[i])
    else:
        appreciation1.append(10 - 0.7 * temperature[i] + 0.02*age[i]+em[i])


OurRegression = np.polyfit(appreciation0,total,1)
print(OurRegression)
summary_across_rows = pd.DataFrame(OurRegression).describe() # across axis=0
print(summary_across_rows)


'''
Exercise Restaurant
'''
n = 1000
Creditcard = list()
sex = [0*i for i in range(n)]
timing = [0*i for i in range(n)]
age = [0*i for i in range(n)]
day = [0*i for i in range(n)]
Creditcard0 = np.random.rand(n)
Creditcard0 = np.round(Creditcard0)

for i in range(len(Creditcard0)):
    if Creditcard0[i] == 0:
        Creditcard.append("Femall")
    else:
        Creditcard.append(("Male"))

spending = 19+stats.chi2.rvs(8,size=n)

for i in range(n):
    if(spending[i]<28):
        sex[i] = int(((np.round(np.random.rand(1))))[0])
        timing[i] = (40+stats.chi2.rvs(8,size=1))[0]
        age[i] = (np.round(25+stats.chi2.rvs(3)))
        day[i] = (1+3* np.round(np.random.rand(1)))[0]
    else:
        sex[i] = 1
        timing[i] = 90 + stats.chi2.rvs(3)
        age[i] = np.round(45+stats.chi2.rvs(6))
        day[i] = np.round(2+2*np.random.rand(1))[0]

for i in range(len(day)):
    if day[i] == 1:
        day[i] = "Mon-Thu"
    elif day[i] == 2:
        day[i] = "Fri"
    elif day[i] == 3:
        day[i] = "Sat"
    elif day[i] == 4:
        day[i] = "Sun"

Resto = pd.DataFrame({'spending':spending,'age':age,'day':day,'sex':sex,'Creditcard':Creditcard,'timing':timing})
print(Resto)

