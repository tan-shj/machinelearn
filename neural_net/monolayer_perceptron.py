import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1],
              [1,0,2]])
#数据标签
Y = np.array([[1],
              [1],
              [-1],
              [-1]])
#初始化权值，3行1列，范围为-1到1
W = (np.random.random([3,1]) - 0.5) * 2

print(W)

#学习率和当前输出
lr = 0.11
O = 0

def update():
    global X,Y,W,lr
    O = np.sign(np.dot(X,W)) #shape:[3,1]
    W_C = (X.T.dot(Y-O))*lr/int(X.shape[0]) #shape:[4,1]
    W = W + W_C 

for i in range(100):
    update()#更新权值
    print(W)
    print(i)
    O = np.sign(np.dot(X,W))#计算当前输出
    if (Y == O).all():  #判断当前输出与样本标签是否完全一致
        print("finished")
        print("epochs:",i)
        break

#正样本
x1 = [3,4]
y1 = [3,3]
#负样本
x2 = [1,0]
y2 = [1,2]

#计算分界线的斜率和偏差
k = -W[1]/W[2]
b = -W[0]/W[2]

xdata = (0,6)

plt.figure()
plt.plot(xdata,xdata*k+b,'r')
plt.scatter(x1,y1,c='b') #(3,3)(4,3)
plt.scatter(x2,y2,c='y') #(1,1)(0,2)
plt.show()
