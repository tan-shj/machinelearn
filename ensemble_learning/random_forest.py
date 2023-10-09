import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

#读取数据
data = np.genfromtxt("\machine_learn\github\classification\LR-testSet2.txt",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

#划分训练集与测试集 默认0.8训练集,0.2测试集
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data)

#画图
def plot(model):
    # 获取数据值所在的范围
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])# ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = z.reshape(xx.shape)
    # 等高线图
    cs = plt.contourf(xx, yy, z)

#建立决策树模型，训练模型，输出分类准确率
Dtree = tree.DecisionTreeClassifier()
Dtree.fit(x_train,y_train)
plot(Dtree)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)#先画等高线图再画散点图才能看得到散点
plt.show()
Dtree_score = Dtree.score(x_test,y_test)
print("Dtree_score:",Dtree_score)

#建立随机森林模型，bagging(又放回的随机抽样)次数为30(决策树的个数)
RF = RandomForestClassifier(n_estimators=30)
RF.fit(x_train,y_train)
plot(RF)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
plt.show()
RF_score = RF.score(x_test,y_test)
print("RF_score:",RF_score)
