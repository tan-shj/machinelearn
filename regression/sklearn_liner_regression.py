# coding=utf-8
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("linear.csv",delimiter=",")
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()

#一维数据->二维 [100,]->[100,1]
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]

#训练模型
model = LinearRegression()
model.fit(x_data,y_data)

#画图
plt.plot(x_data,y_data,'b')
plt.plot(x_data,model.predict(x_data),'r')
plt.show()

