import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import operator
import random

def knn(x_test,x_data,y_data,k):
    # 计算样本数量   6
    x_data_size = x_data.shape[0]
    # 复制x_test,把x_test复制6行1列,计算x_test与每一个样本的差值
    diffMat = np.tile(x_test, (x_data_size,1)) - x_data
    # 计算差值的平方
    sqDiffMat = diffMat**2
    # axis=1,行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances**0.5
    # 从小到大排序,获得下标  [1, 2, 0, 5, 3, 4]
    sortedDistances = distances.argsort()

    classCount = {}
    for i in range(k):
        # 获取标签
        votelabel = y_data[sortedDistances[i]]
        # 统计标签数量 ClassCount.get(votelabel,0):如果没有这个标签，则设置数量为0，有这个标签的话数量为标签数
        classCount[votelabel] = classCount.get(votelabel,0) + 1

    #sorted()为python自带的排序函数，根据key排序，reverse=True时返回降序排列
    #classCount.items()中包含标签和标签数量，operator.itemgetter(1)为标签个数，根据标签个数对classCount排序，然后再取倒序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    # 获取数量最多的标签
    knnclass = sortedClassCount[0][0]
    return knnclass

iris = datasets.load_iris()
#分割数据，0.2为测试数据，0.8为训练数据
#x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)

#打乱数据
data_size = iris.data.shape[0]
index = [i for i in range(data_size)]
random.shuffle(index)
iris.data = iris.data[index]
iris.target = iris.target[index]

#切分数据
test_size = 40
x_train = iris.data[test_size:]
x_test = iris.data[:test_size]
y_train = iris.target[test_size:]
y_test = iris.target[:test_size]

predictions = []
for i in range(x_test.shape[0]):
    predictions.append(knn(x_test[i],x_train,y_train,5))

print(classification_report(y_test,predictions))#分类报告
print(confusion_matrix(y_test,predictions))#混淆矩阵
