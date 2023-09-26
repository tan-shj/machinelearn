import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = np.genfromtxt("/machine_learn/GITHUB/regression/ridge.csv",delimiter=",")
x_data = data[1:,2:]
y_data = data[1:,1]
#print(data)
#print(x_data)
#print(y_data)

alphas_test = np.linspace(0.001,1)#在0.001-1之间均匀的生成50个数，默认50
model = linear_model.RidgeCV(alphas=alphas_test,store_cv_values=True)#训练岭回归模型，输入λ，loss值存在cv.values中
model.fit(x_data,y_data)
print(model.alpha_)
print(model.cv_values_.shape)

plt.plot(alphas_test,model.cv_values_.mean(axis=0))#画出测试λ与loss值的曲线
plt.plot(model.alpha_,min(model.cv_values_.mean(axis=0)),"ro")#用红色的圈标出loss值最低的λ
plt.show()

y_test = model.predict(x_data[2,np.newaxis])
print(y_test)