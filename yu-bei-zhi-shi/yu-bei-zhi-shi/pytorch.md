---
description: Pytorch基础
---

# Pytorch

本节简单介Pytorch的用法

### 为什么需要Pytorch等自动化框架

在上一节中我们学习了梯度下降去寻中一个函数中的最小值，但是前提是我们需要知道梯度。举一个简单的例子，我们可以使用纸和笔计算简单函数的梯度，但是对于深度学模型习来说是不现实的，所以我们需要依赖像Pytorch 、Tensorflow、Tonado框架来提供 自动微分（automatic differentiation）的能力，这样会更简单。

### 使用Pytorch 实现一个简单的神经网络

```text
# PyTorch neural network
# coding=utf-8
import torch
import numpy as np


def nn(x, w1, w2):
    l1 = x @ w1  # 矩阵乘法
    l1 = torch.relu(l1)  # 非线性激活函数
    l2 = l1 @ w2
    return l2


# 权重（参数）矩阵，带有梯度跟踪
w1 = torch.randn(784, 200, requires_grad=True)
w2 = torch.randn(200, 10, requires_grad=True)

# 随机输入向量
x = torch.randn(784, requires_grad=True)

res = nn(x, w1, w2)
print(res)

'''
output:
tensor([ 198.5578, -155.9165,  445.8244,  112.4674,    6.5912, -188.4164,
         473.4200,  169.0547, -109.1408,  -81.7509],
       grad_fn=<SqueezeBackward3>)
'''
```

以上代码与上一节的numpy实现的版本差不多，只不过是使用torch.relu\(\) 函数替代了 np.maximum\(\)，实际上他们实现的功能是一样的，并且我们还向权重矩阵设置添加了 requires\_grad=True 参数，这告诉PyTorch这些是我们想要跟踪梯度的可训练参数，而 x 是一个输入，不是一个可训练的参数。我们还去掉了最后一个激活函数，使之清晰。对于本例，我们将使用著名的MNIST数据集，该数据集包含从0到9的手写数字图像，如图所示。

![&#x6765;&#x81EA;MNIST&#x6570;&#x636E;&#x96C6;&#x7684;&#x624B;&#x7ED8;&#x6570;&#x5B57;&#x7684;&#x793A;&#x4F8B;&#x56FE;&#x50CF;](../../.gitbook/assets/image%20%2825%29.png)

### 使用神经网络分类MNIST\(Pytorch 实现\)

我们想训练我们的神经网络来识别这些图像，并将它们分类为数字0到9。PyTorch有一个相关的库，可以让我们轻松地下载这个数据集。

```text
# Classifying MNIST using a neural network
# coding=utf-8

import torch
import torchvision as TV
import numpy as np


def nn(x, w1, w2):
    l1 = x @ w1  # 矩阵乘法
    l1 = torch.relu(l1)  # 非线性激活函数
    l2 = l1 @ w2
    return l2


# 下载并加载 MNIST 数据集
mnist_data = TV.datasets.MNIST("MNIST", train=True, download=True)

lr = 0.001  # 学习率 （learning rate）

# 进行优化需要的迭代次数
epochs = 2000

batch_size = 100

lossfn = torch.nn.CrossEntropyLoss()  # 建立一个损失函数

# 权重（参数）矩阵，带有梯度跟踪
w1 = torch.randn(784, 200, requires_grad=True)
w2 = torch.randn(200, 10, requires_grad=True)

# 随机输入向量
x = torch.randn(784, requires_grad=True)

res = []

for i in range(epochs):
    # 获取一组随机索引值
    rid = np.random.randint(0, mnist_data.train_data.shape[0], size=batch_size)
    
    # 对数据进行细分并使之变平, 将28 x 28个图像转换成784个向量
    x = mnist_data.train_data[rid].float().flatten(start_dim=1)
    
    # 将向量归一化为0到1之间
    x /= x.max()
    
    # 利用神经网络进行预测
    pred = nn(x, w1, w2)
    
    # 获取实值图像标签
    target = mnist_data.train_labels[rid]
    
    # 计算损失
    loss = lossfn(pred, target)
    
    # 反向传播算法(Backpropagates)
    loss.backward()
    
    # 在这个块不计算梯度
    with torch.no_grad():
        # 参数矩阵上的梯度下降
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
    res.append(loss)

print(res)
```

```text
# 绘图Loss 和 time的关系
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 2000)
y = res
plt.figure(figsize=(20,10))
plt.plot(x,y)
plt.title('Classifying MNIST using a neural network', fontsize=10)
plt.xlabel('time')
plt.ylabel('Loss')
plt.show()
```

![](../../.gitbook/assets/image%20%2828%29.png)

{% hint style="info" %}
以上是我自己跑的结果绘图，实际执行的时候可能和我的结果不一样，原书中的训练结果图如下所示，可以看到曲线不是那么的平滑，训练效果不好，接下使用Adam优化器进行优化
{% endhint %}

 

![](../../.gitbook/assets/image%20%2813%29.png)

通过观察损失函数随着训练时间相当稳定地减少，可以看出神经网络是成功训练的\(图\)。这个简短的代码段训练了一个完整的神经网络，以大约70%的准确率成功地分类MNIST数字。我们只是用和简单对数函数f\(x\) = log\(x 4 + x 3 + 2\)完全一样的方法来实现梯度下降，PyTorch为我们处理了梯度。由于神经网络参数的梯度依赖于输入数据，所以我们每次运行神经网络时都要“向前”运行一个新的图片的随机样本，梯度会有所不同。因此，我们用随机的数据样本向前运行神经网络，PyTorch跟踪发生的计算，当我们完成时，我们对最后的输出调用backward\(\)方法;在这种情况下，通常是损失。backward\(\)方法使用自动微分来计算requires\_grad=True设置的所有PyTorch变量的所有梯度。我们将实际的梯度下降部分封装在torch.no\_grad\(\)上下文中，因为我们不想让它跟踪这些计算。

我们可以通过改进训练算法与一个更复杂的版本梯度下降很容易地达到95%以上的准确率。在下面代码中中，我们简化了我们自己的随机梯度下降的版本，这是随机的部分，因为我们是从数据集随机取子集，并基于此计算梯度，在给定全部数据集的情况下，这给了我们对真实梯度的估计带来影响。

PyTorch包含内置的优化器，其中随机梯度下降\(SGD\)就是其中之一。最流行的替代方案是Adam，它是SGD的一个更复杂的版本。我们只需要用模型参数实例化优化器。

```text
# Classifying MNIST using a neural network
# coding=utf-8

import torch
import torchvision as TV
import numpy as np
import matplotlib.pyplot as plt


def draw(res):
    x = np.arange(0, 2000)
    y = res
    
    # plt.plot(x,y,color='red',marker='o',label='',linewidth=5,markerfacecolor='blue',markersize=5)
    plt.figure(figsize=(20, 10))
    plt.plot(x, y)
    plt.title('Classifying MNIST using a neural network', fontsize=10)
    plt.xlabel('time')
    plt.ylabel('Loss')
    plt.show()


def nn(x, w1, w2):
    l1 = x @ w1  # 矩阵乘法
    l1 = torch.relu(l1)  # 非线性激活函数
    l2 = l1 @ w2
    return l2


# 下载并加载 MNIST 数据集
mnist_data = TV.datasets.MNIST("MNIST", train=True, download=False)

lr = 0.001  # 学习率 （learning rate）

# 进行优化需要的迭代次数
epochs = 2000

batch_size = 100


# 权重（参数）矩阵，带有梯度跟踪
w1 = torch.randn(784, 200, requires_grad=True)
w2 = torch.randn(200, 10, requires_grad=True)

lossfn = torch.nn.CrossEntropyLoss()  # 建立一个损失函数
optim = torch.optim.Adam(params=[w1, w2], lr=lr)

res = []

for i in range(epochs):
    # 获取一组随机索引值
    rid = np.random.randint(0, mnist_data.train_data.shape[0], size=batch_size)
    
    # 对数据进行细分并使之变平, 将28 x 28个图像转换成784个向量
    x = mnist_data.train_data[rid].float().flatten(start_dim=1)
    
    # 将向量归一化为0到1之间
    x /= x.max()
    
    # 利用神经网络进行预测
    pred = nn(x, w1, w2)
    
    # 获取实值图像标签
    target = mnist_data.train_labels[rid]
    
    # 计算损失
    loss = lossfn(pred, target)
    
    # 反向传播算法(Backpropagates)
    loss.backward()

    # 更新参数
    optim.step()
    
    # 重置梯度
    optim.zero_grad()
    
    # # 在这个块不计算梯度
    # with torch.no_grad():
    #     # 参数矩阵上的梯度下降
    #     w1 -= lr * w1.grad
    #     w2 -= lr * w2.grad
    res.append(loss)

if __name__ == '__main__':
    print(res)
    draw(res)
```

![](../../.gitbook/assets/image%20%2822%29.png)

{% hint style="info" %}
以上是我运行时的结果 图。原书中的结果如下所示。
{% endhint %}

![](../../.gitbook/assets/image%20%2815%29.png)

