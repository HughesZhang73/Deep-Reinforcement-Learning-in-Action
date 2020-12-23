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

![&#x6765;&#x81EA;MNIST&#x6570;&#x636E;&#x96C6;&#x7684;&#x624B;&#x7ED8;&#x6570;&#x5B57;&#x7684;&#x793A;&#x4F8B;&#x56FE;&#x50CF;](../../.gitbook/assets/image%20%2822%29.png)

### 使用神经网络分类MNIST\(Pytorch 实现\)

我们想训练我们的神经网络来识别这些图像，并将它们分类为数字0到9。PyTorch有一个相关的库，可以让我们轻松地下载这个数据集。

```text
# Classifying MNIST using a neural network

```



