import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# 定义三个不同的分类器

model = LogisticRegression(max_iter=5000)

scores = model_selection.cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print("5-fold Accuracy: %0.2f [%s]" % (scores.mean(), 'lr'))

model.fit(x_train, y_train)
y_pre = model.predict(x_test)

print('训练集分数：', model.score(x_train, y_train))
print('测试集分数：', model.score(x_test, y_test))

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)