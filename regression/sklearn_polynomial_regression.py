import numpy as np                                      #计算包
import matplotlib.pyplot as plt                         #画图包
from sklearn.linear_model import LinearRegression       #线性回归模型
from sklearn.preprocessing import PolynomialFeatures    #多项式回归模型

#读取数据，取除第一行之外的第一列作为x_data，取除第一行之外的第二列作为y_data，描出图形
data = np.genfromtxt("/machine_learn/GITHUB/regression/job.csv",delimiter=",")#csv结尾的文件分隔符为”，“
x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data,y_data)  #描点
plt.show()

#增加数据的维度 1维->2维 [1,2,3]->[[1],[2],[3]]
x_data = x_data[:,np.newaxis]
y_data = y_data[:,np.newaxis]

#训练线性模型
model = LinearRegression()
model.fit(x_data,y_data)

#用蓝色的点画出测试数据，用红色的线画出预测数据
plt.plot(x_data,y_data,"b.")
plt.plot(x_data,model.predict(x_data),"r")
plt.show()

#使用多项式特征函数(degree=3)转换数据形式，x_poly为转换后的数据，训练模型
#  x   ->    x^0 x^1 x^2 x^3
# [1]       [1 1 1 1]
# [2]  ->   [1 2 4 8]
# [3]       [1 3 9 27]
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x_data)
#print(x_poly)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y_data)

#用蓝色的点画出测试数据，用红色的线画出预测数据
plt.plot(x_data,y_data,"b.")
plt.plot(x_data,lin_reg.predict(x_poly),"r")
plt.title("Polynomial Regression")
plt.plot("Position level")
plt.plot("Salary")
plt.show()
