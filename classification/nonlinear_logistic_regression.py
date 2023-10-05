import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

#数据是否要标准化
scale = False

data = np.genfromtxt("\machine_learn\github\classification\LR-testSet2.txt",delimiter=",")

x_data = data[:,:-1]
y_data = data[:,-1,np.newaxis]

#画点函数
def plot():
    x0=[];y0=[]
    x1=[];y1=[]
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i,0])
            y0.append(x_data[i,1])
        else:
            x1.append(x_data[i,0])
            y1.append(x_data[i,1])
    scatter0 = plt.scatter(x0,y0,c='r',marker='x')
    scatter1 = plt.scatter(x1,y1,c='b',marker='o')
    plt.legend(handles=[scatter0,scatter1],labels=['label0','label1'],loc='best')

#与画点函数效果一样
#plt.scatter(x_data[:,0],x_data[:,1],c=y_data)

plot()
plt.show()

# 定义多项式回归,degree的值可以调节多项式的特征
poly_reg  = PolynomialFeatures(degree=3) 
# 特征处理
x_poly = poly_reg.fit_transform(x_data)

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

    #学习率，迭代次数
    lr = 0.05
    epochs = 50000
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
ws,costlist = gradAscent(x_poly,y_data)
print(ws)

# 获取数据值所在的范围
x_min,x_max = x_data[:,0].min() - 1,x_data[:,0].max() + 1
y_min,y_max = x_data[:,1].min() - 1,x_data[:,1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# np.r_按row来组合array， 
# np.c_按colunm来组合array
# ravel与flatten类似，多维数据转一维。flatten不会直接改变原始数据，ravel会直接改变原始数据
z = sigmoid(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(np.array(ws)))
for i in range(len(z)):
    if z[i] > 0.5:
        z[i] = 1
    else:
        z[i] = 0
z = z.reshape(xx.shape)

# 等高线图
cs = plt.contourf(xx, yy, z)
plot() 
plt.show()

#预测函数
def predict(x_data,ws):
    if scale == True:
        x_data = preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]#使用sigmoid函数分类

predictions = predict(x_poly,ws)
print(classification_report(y_data,predictions))

