# coding=utf-8
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("linear.csv",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]

model = LinearRegression()
model.fit(x_data,y_data)
print("coefficients:",model.coef_)
print("intercept:",model.intercept_)

x_test = [[1024,4]]
predict = model.predict(x_test)
print("predict:",predict)

ax = plt.figure().add_subplot(111,projection='3d')
ax.scatter(x_data[:,0],x_data[:,1],y_data,c='r',marker='o',s=100)
x0 = x_data[:,0]
x1 = x_data[:,1]
x0,x1 = np.meshgrid(x0,x1)#生成网格
z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]

ax.plot_surface(x0,x1,z)
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()
