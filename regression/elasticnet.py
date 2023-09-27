import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = np.genfromtxt("/machine_learn/GITHUB/regression/ridge.csv",delimiter=",")
x_data = data[1:,2:]
y_data = data[1:,1]

#θ^2 -> α*θ^2 + (1-α)*|θ|
model = linear_model.ElasticNetCV()#创建弹性网模型
model.fit(x_data,y_data)#数据维度要一样

print(model.alpha_)#λ
print(model.coef_)#α

#使用弹性网模型预测
y_test = model.predict(x_data)
plt.plot(data[1:,0],y_data,"b.")
plt.plot(data[1:,0],y_test,"r")
plt.show()
