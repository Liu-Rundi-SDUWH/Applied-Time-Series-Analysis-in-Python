'''
MLP
'''
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

y = pd.read_csv("./BAC_0.csv",usecols=[4])
x = pd.read_csv("./BAC_0.csv",usecols=[1,2,3])

x_train = x[0:2997]
x_test = x[2997:]
y_train = y[0:2997]
y_test = y[2997:]

X = x_train
y = y_train

scaler = StandardScaler() # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集
#（多层感知器对特征的缩放是敏感的，所以需要归一化你的数据。 例如，将输入向量 X 的每个属性放缩到到 [0, 1] 或 [-1，+1] ，或者将其标准化使它具有 0 均值和方差 1。
# 为了得到有意义的结果，必须对测试集也应用 相同的尺度缩放。 可以使用 StandardScaler 进行标准化。）
# solver=‘sgd',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(3, 1) hidden层2层,第一层3个神经元，第二层1个神经元，2层隐藏层，也就有3层神经网络
clf = MLPRegressor(solver='sgd', alpha=1e-5,hidden_layer_sizes=(3,1), random_state=1)
clf.fit(X, y)


x_0 = scaler.fit_transform(x_test.loc[2997].values.reshape(1,-1))
print('预测结果：', clf.predict(x_0) ) # 预测某个输入对象
y_predict = clf.predict(x_0)
MSE = (y_test.loc[2997]-y_predict)**2

cengindex = 0
for wi in clf.coefs_:
    cengindex += 1  # 表示底第几层神经网络。
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:',wi.shape)
    print('系数矩阵：\n',wi)

print('pre',y_predict)
print('Test MSE: %.3f' % MSE)

