import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report

#数据是否要标准化
scale = False

data = np.genfromtxt("\machine_learn\github\classification\LR-testSet.csv",delimiter=",")

x_data = data[:,:-1]
y_data = data[:,-1]
#print(len(x_data)) #观察x_data的数据格式

#画点函数
def plot():
    x0 = [];x1 = []
    y0 = [];y1 = []
    
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i,0])#把数据存进x0中
            y0.append(x_data[i,1])
        else:
            x1.append(x_data[i,0])
            y1.append(x_data[i,1])
    scatter0 = plt.scatter(x0,y0,c='b',marker='x')
    scatter1 = plt.scatter(x1,y1,c='r',marker='o')
    plt.legend(handles=[scatter0,scatter1],labels=['lable0','lable1'],loc='best')
    #把scatter0、scatter1放进这个图中，设置标签名称，loc=best自动寻找最合适的位置

plot()
plt.show()

#给数据加偏置，为后面的计算ws做准备
x_data = data[:,:-1]
y_data = data[:,-1,np.newaxis]
print(np.mat(x_data).shape)
print(np.mat(y_data).shape)
X_data = np.concatenate((np.ones((100,1)),x_data),axis=1)
print(X_data.shape)

#定义sigmoid函数，为分类做准备，x大于0时，y大于0.5，归为1类
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#定义损失函数
def cost(xMat,yMat,ws):
    left = np.multiply(yMat,np.log(sigmoid(xMat*ws)))
    right = np.multiply(1-yMat,np.log(1-sigmoid(xMat*ws)))
    return np.sum(left + right) / -(len(xMat))

#梯度下降法
def gradAscent(xArr,yArr):
    if scale == True:
        xArr = preprocessing.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.001
    epochs = 10000
    costlist = []

    m,n = np.shape(xMat)#计算数据行列数，行为样本个数，列为权重值式个数
    ws = np.mat(np.ones((n,1)))#初始化权重值

    for i in range(epochs+1):
        h = sigmoid(xMat*ws)
        ws_grad = xMat.T*(h-yMat)/m
        ws = ws - lr*ws_grad

        if i % 50 == 0:
            costlist.append(cost(xMat,yMat,ws))
    
    return ws,costlist

#训练模型，得到权值和损失值
ws,costlist = gradAscent(X_data,y_data)
print(ws)

if scale == False:
    plot()
    x_test = [[-4],[3]]#取测试数据x的值从-4到3
    y_test = (-ws[0] - x_test * ws[1])/ws[2]#计算预测值
    plt.plot(x_test,y_test,'k')#画出决策边界
    plt.show()

#画图 记录每一次迭代loss值的变化
x = np.linspace(0,10000,201)#0-10000,201次，与上面的迭代数一样
plt.plot(x,costlist,c='r')
plt.title('Train')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

#预测函数
def predict(x_data,ws):
    if scale == True:
        x_data = preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]#使用sigmoid函数分类

predictions = predict(X_data,ws)
print(classification_report(y_data,predictions))

