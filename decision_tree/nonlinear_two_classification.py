import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split

data = np.genfromtxt("\machine_learn\github\classification\LR-testSet2.txt",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]
#print(len(x_data)) #观察x_data的数据格式
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

#分割数据，0.2为测试数据，0.8为训练数据
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)


#创建决策树模型
model = tree.DecisionTreeClassifier(max_depth=7,min_samples_split=7)
#输入数据建立模型
model.fit(x_train,y_train)

# 导出决策树
import graphviz 
dot_data = tree.export_graphviz(model, 
                                out_file = None, 
                                feature_names = ['x','y'],
                                class_names = ['label0','label1'],
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render('nonlinear_two_classification')

# 获取数据值所在的范围
x_min,x_max = x_data[:,0].min() - 1,x_data[:,0].max() + 1
y_min,y_max = x_data[:,1].min() - 1,x_data[:,1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# np.r_按row来组合array， 
# np.c_按colunm来组合array
# ravel与flatten类似，多维数据转一维。flatten不会直接改变原始数据，ravel会直接改变原始数据
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# 等高线图
cs = plt.contourf(xx, yy, z)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

predictions = model.predict(x_train)
print(classification_report(y_train,predictions))

predictions = model.predict(x_test)
print(classification_report(y_test,predictions))
