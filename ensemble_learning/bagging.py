from sklearn import datasets
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#读取鸢尾花数据
iris = datasets.load_iris()
x_data = iris.data[:,:2]
y_data = iris.target

#划分训练集与测试集
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

#knn
#创建模型并训练
knn = neighbors.KNeighborsClassifier()
knn.fit(x_train,y_train)

plot(knn)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
#准确率
knn_score = knn.score(x_test,y_test)
print("knn_score:",knn_score)

#tree
#创建模型并训练
Dtree = tree.DecisionTreeClassifier()
Dtree.fit(x_train,y_train)

plot(Dtree)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
#准确率
Dtree_score = Dtree.score(x_test,y_test)
print("Dtree_score:",Dtree_score)

#bagging_knn
#创建模型并训练
bagging_knn = BaggingClassifier(knn,n_estimators=100)
bagging_knn.fit(x_train,y_train)

plot(bagging_knn)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
#准确率
bagging_knn_score = bagging_knn.score(x_test,y_test)
print("bagging_knn_score:",bagging_knn_score)

#bagging_tree
#创建模型并训练
bagging_Dtree = BaggingClassifier(Dtree,n_estimators=100)
bagging_Dtree.fit(x_train,y_train)

plot(bagging_Dtree)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
#准确率
bagging_Dtree_score = bagging_Dtree.score(x_test,y_test)
print("bagging_Dtree_score:",bagging_Dtree_score)
