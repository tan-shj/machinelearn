import numpy as np
import matplotlib.pyplot as plt
import operator

# 已知分类的数据
x1 = np.array([3,2,1])
y1 = np.array([104,100,81])
x2 = np.array([101,99,98])
y2 = np.array([10,5,2])
scatter1 = plt.scatter(x1,y1,c='r')
scatter2 = plt.scatter(x2,y2,c='b')

# 未知数据
x = np.array([18])
y = np.array([90])
scatter3 = plt.scatter(x,y,c='k')


#画图例
plt.legend(handles=[scatter1,scatter2,scatter3],labels=['labelA','labelB','X'],loc='best')

plt.show()

# 已知分类的数据
x_data = np.array([[3,104],
                   [2,100],
                   [1,81],
                   [101,10],
                   [99,5],
                   [81,2]])
y_data = np.array(['A','A','A','B','B','B'])
x_test = np.array([18,90])

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
# 设置k
k = 5
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
print("X 属于:",knnclass)
