#决策树回归   不常用
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets._california_housing import fetch_california_housing

housing = fetch_california_housing()
#print(housing.DESCR)
print(housing.data.shape)
print(housing.data[0])
print(housing.target[0])

x_data = housing.data
y_data = housing.target
#分割数据
x_train,x_test,y_train,y_test = train_test_split(x_data, y_data)

#创建回归树
model = tree.DecisionTreeRegressor()
model.fit(x_train, y_train)

score = model.score(x_test,y_test)
print(score)#预测准确率