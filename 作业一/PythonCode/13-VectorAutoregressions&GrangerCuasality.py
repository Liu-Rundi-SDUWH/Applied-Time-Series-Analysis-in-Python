'''
Example 1:  Interaction-以“UK_Interest_Rates.xlsx"数据集为例
'''
import statsmodels.api as sm
import statsmodels.stats.diagnostic
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("./Book_dataset/UK_Interest_Rates.xlsx",usecols=[1,2])
R20 = data.R20
RS = data.RS
fig = plt.figure(figsize=(12,8))
plt.plot(R20,'r',label='R20')
plt.plot(RS,'g',label='RS')
plt.title('Correlation: ')
plt.grid()
plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('Price')
plt.show()

adfResult = sm.tsa.stattools.adfuller(R20,3)
output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
					columns=['value'])
output['value']['Test Statistic Value'] = adfResult[0]
output['value']['p-value'] = adfResult[1]
output['value']['Lags Used'] = adfResult[2]
output['value']['Number of Observations Used'] = adfResult[3]
output['value']['Critical Value(1%)'] = adfResult[4]['1%']
output['value']['Critical Value(5%)'] = adfResult[4]['5%']
output['value']['Critical Value(10%)'] = adfResult[4]['10%']

result = sm.tsa.stattools.coint(R20,RS)
print(output)
print(result)

lnDataDict = {'R20':R20,'RS':RS}
lnDataDictSeries = pd.DataFrame(lnDataDict)
data = lnDataDictSeries[['R20','RS']]


orgMod = sm.tsa.VARMAX(data,order=(3,0),exog=None)
fitMod = orgMod.fit(maxiter=1000,disp=False)
print(fitMod.summary())


resid = fitMod.resid
resid = resid['R20']
result = {'fitMod':fitMod,'resid':resid}
result = statsmodels.stats.diagnostic.breaks_cusumolsresid(resid)


ax = fitMod.impulse_responses(2, orthogonalized=True).plot(figsize=(12, 8))
plt.show()