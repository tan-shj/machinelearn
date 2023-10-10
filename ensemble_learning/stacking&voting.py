from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
#投票算法分类器
from sklearn.ensemble import VotingClassifier

FLAG = False

iris = load_iris()

#取第一二列的特征
x_data,y_data = iris.data[:,1:3], iris.target

#定义三个分类器作为输入级
clf1 = LogisticRegression()
clf2 = KNeighborsClassifier(n_neighbors=1)
clf3 = DecisionTreeClassifier()

if FLAG:
    #定义一个输出级分类器
    lr = LogisticRegression()  
    #定义stacking分类器，输入级分类器为clf1,clf2,clf3，输出级分类器为lr
    sclf = StackingClassifier(classifiers=[clf1,clf2,clf3],meta_classifier=lr)
    #循环交叉验证，输出各分类器的准确率
    #zip([],[],[])函数 生成元组 元组内的元素一一对应，迭代到其中一个[]结束
    for clf,lables in zip([clf1,clf2,clf3,sclf],['LogisticRegression','KNN','DecisionTree','stacking']):
        #print(clf,lables)
        #交叉验证，cv=3 1份作测试集，2份作训练集
        scores = model_selection.cross_val_score(clf,x_data,y_data,cv=3)
        print("Accuracy: %0.3f [%s]"%(scores.mean(),lables))

else:
    sclf = VotingClassifier([('LogisticRegression',clf1),('KNN',clf2),('DecisionTree',clf3)])
    for clf,lables in zip([clf1,clf2,clf3,sclf],['LogisticRegression','KNN','DecisionTree','voting']):
        #print(clf,lables)
        #交叉验证，cv=3 1份作测试集，2份作训练集
        scores = model_selection.cross_val_score(clf,x_data,y_data,cv=3)
        print("Accuracy: %0.3f [%s]"%(scores.mean(),lables))

