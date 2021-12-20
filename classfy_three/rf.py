import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# iris_data_set = pd.read_csv('/Users/apple/Desktop/iris_classification_BPNeuralNetwork/sklearn数据集/iris.csv')
data_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/balanced.csv')

# x是4列特征
x = data_set.iloc[:, 0:16].values
# y是1列标签
y = data_set.iloc[:, -1].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 利用GridSearchCV选择最优参数
rf_model = RandomForestClassifier()
param = {'n_estimators': [5, 10, 15, 20, 25]}
grid = GridSearchCV(rf_model, param_grid=param, cv=5)
grid.fit(x_train, y_train)
print('最优分类器:', grid.best_estimator_)
print('最优超参数：', grid.best_params_)
print('最优分数:', grid.best_score_)

# 利用决策树分类器构建分类模型
model = grid.best_estimator_
y_pre = model.predict(x_test)

# print('正确标签：', y_test)
# print('预测结果：', y_pre)

print('训练集分数：', model.score(x_train, y_train))
print('测试集分数：', model.score(x_test, y_test))

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)

# 分类指标文本报告（精确率、召回率、F1值等）
print('分类指标报告：')
print(classification_report(y_test, y_pre))

# 特征重要性
print(model.feature_importances_)

# # 画图展示训练结果
# fig = plt.figure()
# ax = fig.add_subplot(111)
# f1 = ax.scatter(list(range(len(x_test))), y_test, marker='*')
# f2 = ax.scatter(list(range(len(x_test))), y_pre, marker='o')
# plt.legend(handles=[f1, f2], labels=['True', 'Prediction'])
# plt.show()
