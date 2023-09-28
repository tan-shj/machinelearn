import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

scale = False

data = np.genfromtxt("/machine_learn/GIHUB/classification/disabetes.csv",delimiter=",")

x_data = data[1:,:-1]
y_data = data[1:,-1]

def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i,0]) 
            y0.append(x_data[i,1])
        else:
            x1.append(x_data[i,0])
            y1.append(x_data[i,1])
    scatter0 = plt.scatter(x0,y0,c='b',marker='x')
    scatter1 = plt.scatter(x1,y1,c='r',marker='o')
    plt.legend(handles=[scatter0,scatter1],labels=['lable0','lable1'],loc='best')

plot()
plt.show()

x_data = data[1:,:-1]
y_data = data[1:,-1,np.newaxis]
print(np.mat(x_data).shape)
print(np.mat(y_data).shape)
X_data = np.concatenate((np.ones(100,1),x_data),axis=1)
print(X_data.shape)

def sigmoid(x):
    return 1.0/1+np.exp(-x)

def cost(xMat,yMat,ws):
    left = np.multiply(yMat,np.log(sigmoid(xMat*ws)))
    right = np.multiply(1-yMat,np.log(1-sigmoid(xMat*ws)))
    return np.sum(left + right) / -(len(xMat))

def gradAscent(xArr,yArr):
    if scale == True:
        xArr = skl.preprocessing.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.001
    epochs = 10000
    costlist = []

    m,n = np.shape(xMat)
    ws = np.mat(np.ones(n,1))

    for i in range(epochs+1):
        h = sigmoid(xMat*ws)
        ws_grad = xMat.T*(h-yMat)/m
        ws = ws - lr*ws_grad

        if i % 50 == 0:
            costlist.append(cost(xMat,yMat,ws))
    
    return ws,costlist

ws,costlist = gradAscent(X_data,y_data)
print(ws)

if scale == False:
    plot()
    x_test = [[-4],[3]]
    y_test = (-ws[0] - x_test * ws[1])/ws[2]
    plt.plot(x_test,y_test,'k')
    plt.show()

x = np.linspace(0,10000,201)
plt.plot(x,costlist,c='r')
plt.title('Train')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

def predict(x_data,ws):
    if scale == True:
        x_data = skl.preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]

predictions = predict(X_data,ws)
print(skl.classification_report(y_data,predictions))

