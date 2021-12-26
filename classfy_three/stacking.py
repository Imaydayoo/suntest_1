import numpy
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier  # pip install mlxtend
import numpy as np


def convert_weight_to_label(y):
    y_label = np.array(y, int)
    for i in range(len(y)):
        if y[i] <= 2.5:
            y_label[i] = 0
        elif 2.5 < y[i] < 4.0:
            y_label[i] = 1
        else:
            y_label[i] = 2
    return y_label


def regression_test(x_test, y_test_weight, y_pre):
    # 将测试集按照预测标签输入到三个csv文件中
    label0 = []
    label1 = []
    label2 = []
    for i in range(len(y_pre)):
        if y_pre[i] == 0:
            label0.append(i)
        elif y_pre[i] == 1:
            label1.append(i)
        else:
            label2.append(i)

    # 低体重儿分类结果写到pre_lw文件
    lw_data_feature = x_test[label0, :]
    lw_data_label = y_test_weight[label0]
    lw_data = np.c_[lw_data_feature, lw_data_label]
    df_lw = pd.DataFrame(lw_data)
    df_lw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_lw.csv', index=False)

    # 正常体重儿分类结果写到
    nw_data_feature = x_test[label1, :]
    nw_data_label = y_test_weight[label1]
    nw_data = np.c_[nw_data_feature, nw_data_label]
    df_nw = pd.DataFrame(nw_data)
    df_nw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_nw.csv', index=False)

    # 高体重儿分类结果写到
    hw_data_feature = x_test[label2, :]
    hw_data_label = y_test_weight[label2]
    hw_data = np.c_[hw_data_feature, hw_data_label]
    df_hw = pd.DataFrame(hw_data)
    df_hw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_hw.csv', index=False)
    # print(df_hw.shape)


def regression_train(x_train, y_train_weight):
    # 将训练集按照体重划分到三个csv文件中
    label0 = []
    label1 = []
    label2 = []
    for i in range(len(y_train_weight)):
        if y_train_weight[i] <= 2.5:
            label0.append(i)
        elif 2.5 < y_train_weight[i] < 4:
            label1.append(i)
        else:
            label2.append(i)

    # 低体重儿分类结果写到pre_lw文件
    lw_data_feature = x_train[label0, :]
    lw_data_label = y_train_weight[label0]
    lw_data = np.c_[lw_data_feature, lw_data_label]
    df_lw = pd.DataFrame(lw_data)
    df_lw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/train_lw.csv', index=False)

    # 正常体重儿分类结果写到
    nw_data_feature = x_train[label1, :]
    nw_data_label = y_train_weight[label1]
    nw_data = np.c_[nw_data_feature, nw_data_label]
    df_nw = pd.DataFrame(nw_data)
    df_nw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/train_nw.csv', index=False)

    # 高体重儿分类结果写到
    hw_data_feature = x_train[label2, :]
    hw_data_label = y_train_weight[label2]
    hw_data = np.c_[hw_data_feature, hw_data_label]
    df_hw = pd.DataFrame(hw_data)
    df_hw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/train_hw.csv', index=False)
    # print(df_hw.shape)


data_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/balanced.csv')

# x是4列特征
x = data_set.iloc[:, 0:16].values
# 标准化
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std
# y是1列标签
y = data_set.iloc[:, -2].values

# 划分训练集和测试集
x_train, x_test, y_train_weight, y_test_weight = train_test_split(x, y, test_size=0.2, random_state=42)

regression_train(x_train, y_train_weight)

y_train = convert_weight_to_label(y_train_weight)
y_test = convert_weight_to_label(y_test_weight)

# 定义三个不同的分类器
clf1 = KNeighborsClassifier(n_neighbors=12, p=1, weights='distance')
# clf2 = DecisionTreeClassifier()
clf2 = RandomForestClassifier()
clf3 = LogisticRegression(max_iter=5000)
# clf4 = net()

# 定义一个次级分类器
lr = LogisticRegression(max_iter=5000)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN', 'Decision Tree', 'LogisticRegression', 'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, x_train, y_train, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))

sclf.fit(x_train, y_train)

y_pre = sclf.predict(x_test)

print('训练集分数：', sclf.score(x_train, y_train))
print('测试集分数：', sclf.score(x_test, y_test))

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)

regression_test(x_test, y_test_weight, y_pre)
# # 将测试集按照预测标签输入到三个csv文件中
# label0 = []
# label1 = []
# label2 = []
# for i in range(len(y_pre)):
#     if y_pre[i] == 0:
#         label0.append(i)
#     elif y_pre[i] == 1:
#         label1.append(i)
#     else:
#         label2.append(i)
#
# # 低体重儿分类结果写到pre_lw文件
# lw_data_feature = x_test[label0, :]
# lw_data_label = y_test_weight[label0]
# lw_data = np.c_[lw_data_feature, lw_data_label]
# df_lw = pd.DataFrame(lw_data)
# df_lw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_lw.csv', index=False)
#
# # 正常体重儿分类结果写到
# nw_data_feature = x_test[label1, :]
# nw_data_label = y_test_weight[label1]
# nw_data = np.c_[nw_data_feature, nw_data_label]
# df_nw = pd.DataFrame(nw_data)
# df_nw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_nw.csv', index=False)
#
# # 高体重儿分类结果写到
# hw_data_feature = x_test[label2, :]
# hw_data_label = y_test_weight[label2]
# hw_data = np.c_[hw_data_feature, hw_data_label]
# df_hw = pd.DataFrame(hw_data)
# df_hw.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_hw.csv', index=False)
# # print(df_hw.shape)
