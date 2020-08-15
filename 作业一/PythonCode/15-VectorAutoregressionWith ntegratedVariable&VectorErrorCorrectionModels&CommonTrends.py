'''
Example 2:  VECM Representation-以“UK_Interest_Rates.xlsx"数据集为例
'''
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
df = pd.read_excel('./Book_dataset/UK_Interest_Rates.xlsx', usecols=[1,2])
grangercausalitytests(df, maxlag=3)

'''
Example 3: Tests on the VECM-以“USB.csv"数据集为例
'''
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
df = pd.read_csv('./Book_dataset/USB.csv', usecols=[1,2])
grangercausalitytests(df, maxlag=3)