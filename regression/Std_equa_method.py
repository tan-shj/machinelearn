import numpy as np
from matplotlib import pyplot as plt

#使用的数据是随便给的
#导入数据，取数据的同时给x_data、y_data增加一个维度 向量->矩阵 用散点图表示出来
data = np.genfromtxt("/machine_learn/GITHUB/regression/job.csv",delimiter=",")
x_data = data[1:,1,np.newaxis]
y_data = data[1:,2,np.newaxis]
#print(x_data)
#print(data[1:,1])
plt.scatter(x_data,y_data)
plt.show()

#ones(10,1)生成一个10行1列的矩阵与x_data连接起来，axis=1(0)：方式为把列(行)并起来
print(np.mat(x_data).shape)
X_data = np.concatenate((np.ones((10,1)),x_data),axis=1)
print(X_data.shape)

#定义一个函数计算特征的权重 W=(X.T*X).I*X.T*Y
def weights(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat
    if(np.linalg.det(xTx) == 0.0):
        print("This Matrix cannot to inverse")
        print("这个矩阵不能取反,即不可逆")
        return
    ws = xTx.I*xMat.T*yMat
    return ws

#计算x_data中x0，x1的权重
ws = weights(X_data,y_data)
print(ws)

#测试数据为两行一列的矩阵[[2],[8]] 
x_test = np.array([[2],[8]])
print(x_test)
print(x_test.shape)
#通过x0，x1的权重计算预测值
y_test = ws[0] + x_test*ws[1]
print(y_test)
plt.plot(x_data,y_data,"b.")
plt.plot(x_test,y_test,"r")
plt.show()

