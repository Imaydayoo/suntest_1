import torch
import torch.nn.functional as F  # 包含激励函数
import matplotlib.pyplot as plt

# 假数据
# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torctenh.ones(100)  # class1 y data (tensor), shape=(100, 1)
# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
# 画散点图
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


# 建立神经网络
# 先定义所有的层属性(__init__()), 然后再一层层搭建(forward(x))层于层的关系链接
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)  # define the network
print(net)  # net architecture  == 显示神经网络结构
# Net(
#   (hidden): Linear(in_features=2, out_features=10, bias=True)
#   (out): Linear(in_features=10, out_features=2, bias=True)
# )
# 搭建完神经网络后，对 神经网路参数（net.parameters()） 进行优化
# (1.选择优化器 optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# (2.选择优化的目标函数
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()  # something about plotting
# (3.开始训练网络
for t in range(100):
    out = net(x)  # input x and predict based on x  # 喂给 net 训练数据 x, 输出预测值
    loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted  # 计算两者的误差

    optimizer.zero_grad()  # clear gradients for next train      # 清空上一步的残余更新参数值
    loss.backward()  # backpropagation, compute gradients  # 误差反向传播, 计算参数更新值
    optimizer.step()  # apply gradients                     # 将参数更新值施加到 net 的 parameters 上

    if t % 2 == 0:
        # plot and show learning process
        # plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)  # 预测中有多少和真实值一样
        print('准确率：', accuracy)
        # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        # plt.pause(0.1)

# plt.ioff()
# plt.show()
