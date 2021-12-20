import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset


# def get_accuracy(y_pre, y_label):
#     prediction = torch.max(y_pre, 1)[1]
#     pred_y = prediction.data.numpy()
#     target_y = y_label.data.numpy()
#     # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#     accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)  # 预测中有多少和真实值一样
#     return accuracy

def get_accuracy(y_pred, y_true):
    y_pred = y_pred.data.numpy()
    y_true = y_true.data.numpy()
    count = 0
    for (yt, yp) in zip(y_true, y_pred):
        if np.abs(yt - yp) < 0.25:
            count += 1
    return count / len(y_true)


'''
使用正态分布随机生成两类数据
第一类有100个点，使用均值为2，标准差为1的正态分布随机生成，标签为0。
第二类有100个点，使用均值为-2，标准差为1的正态分布随机生成，标签为1。
torch.normal(tensor1,tensor2)
输入两个张量，tensor1为正态分布的均值，tensor2为正态分布的标准差。
torch.normal以此抽取tensor1和tensor2中对应位置的元素值构造对应的正态分布以随机生成数据，返回数据张量。
'''

# x1_t = torch.normal(2 * torch.ones(100, 2), 1)
# y1_t = torch.zeros(100)
#
# x2_t = torch.normal(-2 * torch.ones(100, 2), 1)
# y2_t = torch.ones(100)
#
# x_t = torch.cat((x1_t, x2_t), 0)
# y_t = torch.cat((y1_t, y2_t), 0)

data_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/balanced.csv')

# x是4列特征
x = data_set.iloc[:, 0:16].values

# 特征正则化
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std

# y是1列标签
y = data_set.iloc[:, -2].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# y_train_three = get_label(y_train).T
x_train_data = torch.from_numpy(x_train)
y_train_data = torch.from_numpy(y_train)
x_train_data = x_train_data.type(torch.FloatTensor)
y_train_data = y_train_data.type(torch.FloatTensor)
deal_dataset = TensorDataset(x_train_data, y_train_data)
train_loader = DataLoader(dataset=deal_dataset,
                          batch_size=100,
                          shuffle=True)

'''
搭建神经网络，
输入层包括2个节点，两个隐层均包含5个节点，输出层包括1个节点。
'''

net = nn.Sequential(
    nn.Linear(16, 6),  # 输入层与第一隐层结点数设置，全连接结构
    torch.nn.ReLU(),  # 第一隐层激活函数采用sigmoid
    # nn.Linear(12, 6),  # 第一隐层与第二隐层结点数设置，全连接结构
    # torch.nn.Tanh(),  # 第一隐层激活函数采用sigmoid
    # nn.Linear(10, 5),  # 第二隐层与第三隐层结点数设置，全连接结构
    # torch.nn.Tanh(),  # 第二隐层激活函数采用
    nn.Linear(6, 1),  # 第三隐层与输出层层结点数设置，全连接结构
    # nn.Softmax(dim=1)  # 由于有两个概率输出，因此对其使用Softmax进行概率归一化
)

# print(net)
'''
Sequential(
  (0): Linear(in_features=2, out_features=5, bias=True)
  (1): Sigmoid()
  (2): Linear(in_features=5, out_features=5, bias=True)
  (3): Sigmoid()
  (4): Linear(in_features=5, out_features=2, bias=True)
  (5): Softmax(dim=1)
)'''

# 配置损失函数和优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.15)  # 优化器使用随机梯度下降，传入网络参数和学习率
# loss_func = torch.nn.CrossEntropyLoss()  # 损失函数使用交叉熵损失函数
loss_func = torch.nn.L1Loss()

# 模型训练
num_epoch = 1001  # 最大迭代更新次数
for epoch in range(num_epoch):
    acc = 0
    for _, data in enumerate(train_loader):
        batch_data, batch_label = data
        y_p = net(batch_data)  # 喂数据并前向传播
        batch_label = batch_label.float()
        # batch_label = batch_label.re
        y_p = y_p.squeeze(-1)
        loss = loss_func(y_p, batch_label)  # 计算损失
        '''
        PyTorch默认会对梯度进行累加，因此为了不使得之前计算的梯度影响到当前计算，需要手动清除梯度。
        pyTorch这样子设置也有许多好处，但是由于个人能力，还没完全弄懂。
        '''
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 计算梯度，误差回传
        optimizer.step()  # 根据计算的梯度，更新网络中的参数

        acc = get_accuracy(y_p, batch_label)

    if epoch % 100 == 0:
        print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, loss.data.item(), acc))

'''
torch.max(y_p,dim = 1)[0]是每行最大的值
torch.max(y_p,dim = 1)[0]是每行最大的值的下标，可认为标签
'''
# print("所有样本的预测标签: \n", torch.max(y_p, dim=1)[1])

x_test_data = torch.from_numpy(x_test)
y_test_data = torch.from_numpy(y_test)
x_test_data = x_test_data.type(torch.FloatTensor)
y_test_data = y_test_data.type(torch.LongTensor)
y_pre = net(x_test_data)

accuracy = get_accuracy(y_pre, y_test_data)
print('准确率：', accuracy)
