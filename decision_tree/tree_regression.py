import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

data = np.genfromtxt("\machine_learn\github\decision_tree\data.csv",delimiter=",")
x_data = data[:,0,np.newaxis]
y_data = data[:,-1,np.newaxis]

plt.scatter(x_data,y_data)
plt.show()

#创建决策树模型
model = tree.DecisionTreeRegressor(max_depth=4)
#输入数据建立模型
model.fit(x_data,y_data)

x_test = np.linspace(20,80,100)
x_test = x_test[:,np.newaxis]

# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_test, model.predict(x_test), 'r')
plt.show()

# 导出决策树
import graphviz # http://www.graphviz.org/

dot_data = tree.export_graphviz(model, 
                                out_file = None, 
                                feature_names = ['x'],
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render('tree_regression')
