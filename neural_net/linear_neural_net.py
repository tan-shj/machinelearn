#线性神经网络异或问题

import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,0,0,0,0,0],
              [1,0,1,0,0,1],
              [1,1,0,1,0,0],
              [1,1,1,1,1,1]])
#数据标签
Y = np.array([[-1],
              [1],
              [1],
              [-1]])
#初始化权值，6行1列，范围为-1到1
W = (np.random.random([6,1]) - 0.5) * 2

print(W)

#学习率和当前输出
lr = 0.11
O = 0

def update():
    global X,Y,W,lr
    O = np.dot(X,W) #shape:[4,1]
    W_C = (X.T.dot(Y-O))*lr/int(X.shape[0]) #shape:[6,1]
    W = W + W_C 

#三种方法：设置一定迭代次数  预测值与真实值的误差在一定范围内   权值在一定范围内浮动
for i in range(100):
    update()#更新权值
   
#正样本
x1 = [0,1]
y1 = [1,0]
#负样本
x2 = [0,1]
y2 = [0,1]

#解一元二次方程，得y
def caculate(x,root):
    a = W[5]
    b = W[2]+x*W[4]
    c = W[0]+x*W[1]+x*x*W[3]
    if root == 1:
        return (-b+np.sqrt(b*b-4*a*c))/(2*a)
    else:
        return (-b-np.sqrt(b*b-4*a*c))/(2*a)

#均匀的生成-1到2之间的num个数，默认50
xdata = np.linspace(-1,2)
#print(len(xdata))

plt.figure()
plt.plot(xdata,caculate(xdata,1),'r')
plt.plot(xdata,caculate(xdata,2),'r')
plt.plot(x1,y1,'bo') #(0,1)(1,0)
plt.plot(x2,y2,'yo') #(0,0)(1,1)
plt.show()