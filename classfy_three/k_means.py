import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sklearn.datasets as ds

# 数据集的个数
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data_num = 1000
# k值，同时也是生成数据集的中心点
k_num = 3

# 生成聚类数据，默认n_features为二，代表二维数据，centers表示生成数据集的中心点个数，random_state随机种子
data, y = ds.make_blobs(n_samples=data_num, centers=k_num, random_state=0)

data_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/balanced.csv')

# x是4列特征
x = data_set.iloc[:, 0:16].values
# y是1列标签
y = data_set.iloc[:, -1].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

KMeans(
    n_clusters=8,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm='auto')
model = KMeans(n_clusters=k_num, init='k-means++')

# 训练
model.fit(x_train)

# 分类预测
y_pre = model.predict(x_test)

conf_mat = confusion_matrix(y_test, y_pre)
print('混淆矩阵：')
print(conf_mat)

# data_colors = matplotlib.colors.ListedColormap(['red', 'blue', 'yellow'])
# plt.scatter(data[:, 0], data[:, 1], c=y_pre, cmap=data_colors)
# plt.title("k-means' result")
# plt.grid()
# plt.show()
