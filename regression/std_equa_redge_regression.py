import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = np.genfromtxt("/machine_learn/GITHUB/regression/ridge.csv",delimiter=",")
x_data = data[1:,2:]
y_data = data[1:,1,np.newaxis]

#ones(16,1)生成一个16行1列的矩阵与x_data连接起来，axis=1(0)：方式为把列(行)并起来
X_data = np.concatenate((np.ones((16,1)),x_data),axis=1)

alphas_test = np.linspace(0.001,1)#在0.001-1之间均匀的生成50个数，默认50
model = linear_model.RidgeCV(alphas=alphas_test,store_cv_values=True)#训练岭回归模型，输入λ，loss值存在cv.values中
model.fit(x_data,data[1:,1])

#用岭回归的loss函数定义一个函数计算特征的权重 W=(X.T*X).I*X.T*Y
def weights(xArr,yArr,lam=model.alpha_):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat
    rxTx = xTx + np.eye(xMat.shape[1])*lam
    if(np.linalg.det(rxTx) == 0.0):
        print("This Matrix cannot to inverse")
        print("这个矩阵不能取反,即不可逆")
        return
    ws = rxTx.I*xMat.T*yMat
    return ws

#计算x_data中各分量的权重
ws = weights(X_data,y_data)
print(ws)

y_test = np.mat(X_data)*np.mat(ws)
print(y_test)