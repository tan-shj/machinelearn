import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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

logistic = LogisticRegression()
logistic.fit(x_data,y_data)

print(logistic.coef_.shape)#1行2列
print(logistic.coef_)#[[ 0.85767013 -1.54232428]] -> 2维
print(logistic.coef_[0])#[ 0.85767013 -1.54232428] -> 1维

if scale == False:
    plot()
    x_test = np.array([[-4],[3]])#取测试数据x的值从-4到3
    y_test = (-logistic.intercept_ - x_test * logistic.coef_[0][0])/logistic.coef_[0][1]#计算预测值
    plt.plot(x_test,y_test,'k')#画出决策边界
    plt.show()

predictions = logistic.predict(x_data)
print(classification_report(y_data,predictions))
