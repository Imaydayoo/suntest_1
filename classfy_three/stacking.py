import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier # pip install mlxtend
import numpy as np

# 载入数据集
# iris = datasets.load_iris()
# # 只要第1,2列的特征
# x_data, y_data = iris.data[:, 1:3], iris.target

data_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/balanced.csv')

# x是4列特征
x = data_set.iloc[:, 0:16].values
# 标准化
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std
# y是1列标签
y = data_set.iloc[:, -1].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义三个不同的分类器
clf1 = KNeighborsClassifier(n_neighbors=12, p=1, weights='distance')
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression(max_iter=5000)
clf4 = net()

# 定义一个次级分类器
lr = LogisticRegression(max_iter=5000)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN', 'Decision Tree', 'LogisticRegression', 'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, x_train, y_train, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))