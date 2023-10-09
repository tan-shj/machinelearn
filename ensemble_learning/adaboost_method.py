import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np

#生成二维正态分布数据，样本数500，默认均值(0,0)，2个特征，分为2类
x1,y1 = make_gaussian_quantiles(n_samples=500,n_features=2,n_classes=2)
#生成二维正态分布数据，样本数500，均值(3,3)，2个特征，分为2类
x2,y2 = make_gaussian_quantiles(mean=(3,3),n_samples=500,n_features=2,n_classes=2)

#将两组数据合成一组数据，这样生成的数据是打乱的
x_data = np.concatenate((x1,x2))
y_data = np.concatenate((y1,-y2+1))#2类 0,1  -y2+1  (0,1)->(1,0)

# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

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

#单纯决策树分类  决策树深度3
dtree = tree.DecisionTreeClassifier(max_depth=3)
dtree.fit(x_data,y_data)
plot(dtree)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
tree_score = dtree.score(x_data,y_data)
print("tree_score",tree_score)

#使用adaboost算法配合决策树分类   分类方法为决策树，迭代次数10
adaboost = AdaBoostClassifier(dtree,n_estimators=10)
adaboost.fit(x_data,y_data)
plot(adaboost)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
adaboost_score = adaboost.score(x_data,y_data)
print("adaboost_score",adaboost_score)
