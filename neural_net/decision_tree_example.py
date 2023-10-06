from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import csv

#读入数据
Dtree = open(r'\machine_learn\github\neural_net\AllElectronics.csv','r')
reader = csv.reader(Dtree)

#获取第一行数据
headers = reader.__next__()
print(headers)

#定义两个列表
featureList = []
labelList = []

for row in reader:
    #把label存入list
    #print(row[-1]) 标签
    #print(headers[1])
    labelList.append(row[-1])
    rowDict = {}
    #len(row) = 6,从第一列开始，不包括RID
    for i in range(1,len(row)-1):
        #建立一个数据字典
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict) 

print(featureList)

#把数据转换成01表示
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()

#打印转化后的数据
print("x_data:" + str(x_data))
#打印属性名称
print(vec.get_feature_names_out())
#打印标签
print("labelList:" + str(labelList))

#把标签转换成01表示
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_data" + str(y_data))

#创建决策树模型
model = tree.DecisionTreeClassifier(criterion='entropy')
#输入数据建立模型
model.fit(x_data,y_data)

#测试
x_test = x_data[0]
print("x_test:" + str(x_test))

# d = array(mat).reshape(m,-1) 
# array(mat).shape=(a,b) d.shape=(m,a*b/m)
predict = model.predict(x_test.reshape(1,-1))
print("predict:" + str(predict))

# 导出决策树
import graphviz 
dot_data = tree.export_graphviz(model, 
                                out_file = None, 
                                feature_names = vec.get_feature_names_out(),
                                class_names = lb.classes_,
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render('computer')

graph
print(vec.get_feature_names_out())
print(lb.classes_)

