import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from torch.utils.data import DataLoader, TensorDataset


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


# data_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_lw.csv')
train_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/train_nw.csv')
test_val_set = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/pre_nw.csv')
x_train = train_set.iloc[:, 0:16].values
y_train = train_set.iloc[:, -1].values
x_test_val = test_val_set.iloc[:, 0:16].values
y_test_val = test_val_set.iloc[:, -1].values
# # x是4列特征
# x = data_set.iloc[:, 0:16].values

# # 特征正则化
# mean = x.mean(axis=0)
# std = x.std(axis=0)
# x = (x - mean) / std

# # y是1列标签
# y = data_set.iloc[:, -2].values

# 划分训练集和测试集
# x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size=0.2, random_state=2)
x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=42)

x_val_tensor = torch.from_numpy(x_val)
y_val_tensor = torch.from_numpy(y_val)
x_val_tensor = x_val_tensor.type(torch.FloatTensor)
y_val_tensor = y_val_tensor.type(torch.FloatTensor)
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
    nn.Linear(16, 12),  # 输入层与第一隐层结点数设置，全连接结构
    torch.nn.ReLU(),  # 第一隐层激活函数采用sigmoid
    nn.Linear(12, 10),  # 第一隐层与第二隐层结点数设置，全连接结构
    torch.nn.Tanh(),  # 第一隐层激活函数采用sigmoid
    nn.Linear(10, 5),  # 第二隐层与第三隐层结点数设置，全连接结构
    torch.nn.Tanh(),  # 第二隐层激活函数采用
    nn.Linear(5, 1),  # 第三隐层与输出层层结点数设置，全连接结构
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
optimizer = torch.optim.SGD(net.parameters(), lr=0.07)  # 优化器使用随机梯度下降，传入网络参数和学习率
# loss_func = torch.nn.CrossEntropyLoss()  # 损失函数使用交叉熵损失函数
loss_func = torch.nn.MSELoss()

# 模型训练
num_epoch = 1001  # 最大迭代更新次数
for epoch in range(num_epoch):
    acc_val_best = 0
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

    y_train_pre = net(x_train_data)
    y_val_pre = net(x_val_tensor)
    acc_train = get_accuracy(y_train_pre, y_train_data)
    acc_val = get_accuracy(y_val_pre, y_val_tensor)
    if acc_val >= acc_val_best:
        torch.save(net, 'net.pkl')
    if epoch % 100 == 0:
        print('epoch: {}, loss: {}, acc_train: {}, acc_val: {}'.format(epoch, loss.data.item(), acc_train, acc_val))

'''
torch.max(y_p,dim = 1)[0]是每行最大的值
torch.max(y_p,dim = 1)[0]是每行最大的值的下标，可认为标签
'''
# print("所有样本的预测标签: \n", torch.max(y_p, dim=1)[1])

x_test_tensor = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)
# x_test_tensor = x_test_data.type(torch.FloatTensor)
# y_test_tensor = y_test_data.type(torch.LongTensor)

net = torch.load('net.pkl')
y_pre = net(x_test_tensor)

accuracy = get_accuracy(y_pre, y_test_tensor)
print('准确率：', accuracy)
print(len(y_pre))
