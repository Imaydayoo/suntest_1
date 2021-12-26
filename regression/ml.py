import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR, NuSVR

df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/train_nw.csv')
df_hw = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_nw.csv')
print(df.columns)

# 相关性分析
# print(df.corr(method='spearman'))

x = df.iloc[:, 0:16]
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std

y = df.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
x_test = df_hw.iloc[:, 0:-1]
y_test = df_hw.iloc[:, -1]
# 建立模型

#   1 线性模型
# model = LinearRegression()
# model.fit(x_train, y_train)
# y_test_pred = model.predict(x_test)


# # 2 多项式回归
# poly_reg = PolynomialFeatures(degree=2)  # 2最好
# x_poly = poly_reg.fit_transform(x)
# model = LinearRegression()
# model.fit(x, y)
# y_test_pred = model.predict(poly_reg.transform(x_test))


# 3 支持向量机SVM回归 卡住了
# model = SVR(kernel='linear', gamma=0.01,  C=1, epsilon=0.2)


# 4 支持向量机SVM回归 卡住了 原来是时间久一些
# model = NuSVR(kernel='rbf', nu=0.5, gamma=0.01, C=1)
# model.fit(x_train, y_train)
# y_test_pred = model.predict(x_test)


# # 5 使用KNN回归算法求解
model = KNeighborsRegressor(n_neighbors=10)
model.fit(x, y)
y_test_pred = model.predict(x_test)


# 传统公式法

# 法一
# y_test_pred = np.power(10, 1.5662 - 0.0108 * x_test['HC'] / 10 + 0.0468 * x_test['FAC'] / 10 + 0.171 * x_test['FL'] / 10
#                        + 0.00034 * np.power(x_test['HC'], 2) / 100 - 0.003685 * x_test['FAC'] * x_test['FL'] / 100) \
#                         / 1000
# y_test_pred = -1.7492 - 0.166 * x_test['BPD'] + 0.046 * x_test['FAC'] - 2.646 * x_test['MAC'] * x_test['BPD'] / 1000
# print(np.power(10,y_test_pred))

# 公式法二
# y_test_pred = np.power(10, 1.3596 - 0.00386 * x_test['FAC'] * x_test['FL'] / 100 + 0.0064 * x_test['HC'] / 10
#                        + 0.00061 * x_test['BPD'] * x_test['FAC'] / 100 + 0.0424 * x_test['FAC'] / 10 + 0.174 *
#                        x_test['FL'] / 10) / 1000

# y_test_pred = -1.7492 - 0.166 * x_test['BPD'] + 0.046 * x_test['FAC'] - 2.646 * x_test['MAC'] * x_test['BPD'] / 1000
# print(np.power(10,y_test_pred))


# + 0.0468 * x_test['MAC'] + 0.171 * x_test['FL'] + \
# 0.00034 * x_test['HC'] - 0.003685 * x_test['MAC'] * x_test['FL']
# y_test_pred = []
# for i in range(len(x_test))
#     fw = 1.5662 - 0.0108 * x_test[i]['age'] + 0.0468
#          # * x_test[i][4] + 0.171 * x_test[i][8] + 0.00034 * x_test[i][7] * x_test[i][7] - 0.003685 * x_test[i][4] * x_test[i][8]
#     y_test_pred.append(fw)


# 评价模型
# 2 MAPE
def mpe(y_true, y_pred):
    return np.mean((y_pred - y_true) / y_true) * 100


# 2 MAPE
def mape(y_true, y_pred):
    temp = np.abs(y_pred - y_true) / y_true
    return np.mean(temp)
    # return np.mean(np.abs(y_pred - y_true) / y_true) * 100


# 3 smape 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100


# 4 准确率
def accuracy(y_true, y_pred):
    count = 0
    for (yt, yp) in zip(y_true, y_pred):
        if np.abs(yt - yp) < 0.25:
            count += 1
    return count / len(y_true)


# y_test = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
# y_test_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])

# MSE
print("MSE =", metrics.mean_squared_error(y_test, y_test_pred))

# RMSE
print("RMSE =", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

# MAE
print("MAE =", metrics.mean_absolute_error(y_test, y_test_pred))

# MAPE
print("MAPE =", mape(y_test, y_test_pred))

# SMAPE
print("SMAPE =", smape(y_test, y_test_pred))

# 准确率
print("准确率 =", accuracy(y_test, y_test_pred))

# test_score = r2_score(y_test, y_test_pred)
# print("R^2得分=", test_score)
