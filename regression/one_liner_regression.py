# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
###
#载入数据，第一列为x_Data，第二列为y_data
data = np.genfromtxt("linear.csv",delimiter=",")
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()

lr = 0.0001     #学习率learning rate
b = 0           #截距
k = 0           #斜率
epochs = 50     #最大迭代次数

#最小二乘法，计算误差函数  (x ** n):求x^n
def compute_error(b,k,x_data,y_data):
    totalError = 0
    for i in range(0,len(x_data)):
        totalError += (y_data[i]-(k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data)) / 2.0

def gradient_descent_runner(x_data,y_data,b,k,lr,epochs):
    #计算总数据量
    m = float(len(x_data))
    #循环epochs次
    for i in range(0,epochs):
        b_grad = 0
        k_grad = 0
        #计算梯度的总和再求平均
        for j in range(0,len(x_data)):
            b_grad += (1/m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1/m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
        #更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
        #每迭代5次输出一次图像
        # if i % 5 == 0:
        #     print("epochs:",i)
        #     plt.plot(x_data,y_data,'b')
        #     plt.plot(x_data,k*x_data+b,'r')
        #     plt.show()
    return b,k

print("Starting b = {0} , k = {1} , error = {2}".format(b,k,compute_error(b,k,x_data,y_data)))
print("Running...")
b , k = gradient_descent_runner(x_data,y_data,b,k,lr,epochs)
print("After {0}  iterations b = {1} , k = {2} , error = {3}".format(epochs,b,k,compute_error(b,k,x_data,y_data)))

#画图
plt.plot(x_data,y_data,'b')
plt.plot(x_data,k*x_data+b,'r')
plt.show()




