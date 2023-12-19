<link rel="stylesheet" href="../css/counter.css" />

# HW4: Learning CNN

!!! warning "仅 LeNet-5 相关内容设计完成"

## 实验简介

- **深度学习**（Deep Learning）：[机器学习](https://zh.wikipedia.org/wiki/机器学习)的分支，是一种以[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络)为架构，对数据进行表征学习的[算法](https://zh.wikipedia.org/wiki/算法)
- **卷积神经网络**（Convolutional Neural Network, **CNN**）：一种[前馈神经网络](https://zh.wikipedia.org/wiki/前馈神经网络)，对于大型图像处理有出色表现

本次实验我们将完成

1. LeNet-5 的训练，应用于 MNIST 数据集上的手写数字识别任务（图像分类）
2. （待定）UNet 的网络补全，应用于 [Carvana 数据集](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)上的语义分割任务
3. （待定）FRCN 的网络补全，应用于 [NYU Depth 数据集](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)上的单目深度估计任务

## 实验环境

要求使用 python + pytorch 完成实验，推荐使用 miniconda 或者 anaconda 管理环境。

### 安装 conda

用 conda 管理环境是因为你可能要用 python 完成多个项目，有的项目的环境之间可能存在难以解决甚至无法解决的冲突。conda 可以帮助你在同一台机子的同一个账户下创建和管理多个 python 环境，各个环境相互独立，不会相互影响；而且每个环境封装在一个文件夹中，克隆、移除都很方便。

Anaconda 完全包含了 miniconda，预装了许多内容，也提供图形化功能。相比之下 miniconda 比较轻量级，只提供 python 和 conda 功能。相较之下更推荐使用 miniconda，可以在 [Latest Miniconda installer links by Python version](https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html) 下载安装最新版的 miniconda。

Windows 的 conda 环境配置比较麻烦，需要大家各自搜索解决环境问题，可以适当参考[这篇知乎文章](https://zhuanlan.zhihu.com/p/591091259)。另一种解决方案是使用 WSL，如果需要可以参考本人在另一门课程写的[教程](https://zhoutimemachine.github.io/2023_FPA/env/windows_lost/#wsl)进行安装。

### 新建 conda 环境

conda 默认环境为 base，如果做什么都使用这个环境，很容易导致环境混乱。

先为自己要使用的环境起一个名字，例如 cv。下面的命令可以创建一个名为 cv，python 版本为 3.10 的环境。

```bash
conda create cv python=3.10
```

conda 默认环境为 base，需要激活新建的环境才能使用。

```bash
conda activate cv
```

随后就可以使用 conda 或者 pip 安装所需要的包了。例如 Pytorch，可以通过 [PyTorch 官网](https://pytorch.org/)找到对应的安装命令。

例如，Stable (2.1.2) - Windows - Pip - Python - CPU 对应的 Pytorch 安装命令为

```bash
pip3 install torch torchvision torchaudio
```

## 实验基础知识介绍

### 网络模型

#### CNN

CNN 由一个或多个卷积层和末尾的全连接层（对应经典的神经网络）组成，同时也包括关联权重和池化层（pooling layer）。

CNN 的特点包括

- **局部连接**：卷积层的输出中的单个元素只取决于输入 feature map 中的局部区域
- **权值共享**：在输入 feature map 的不同位置使用相同的参数（同一卷积核）

这些特点使其参数量大大减少，且对局部的空间特征具有很好的提取作用。

与其他深度学习结构相比，卷积神经网络在图像和[语音识别](https://zh.wikipedia.org/wiki/语音识别)方面能够给出更好的结果。这一模型也可以使用[反向传播算法](https://zh.wikipedia.org/wiki/反向传播算法)进行训练，相比较其他深度、前馈神经网络，卷积神经网络需要考量的参数更少，使之成为一种颇具吸引力的深度学习结构。

#### LeNet-5

LeNet-5 是一个简单的经典 CNN，可以称之为 CNN 中的 "Hello World"。下图显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后输出每种分类预测得到的概率。

<div style="text-align:center;">
<img src="../graph/LeNet.jpg" alt="How to Train a Model with MNIST dataset | by Abdullah Furkan Özbek | Medium" style="margin: 0 auto;"/>
</div>

有关于其更详细的结构可以在 LeNet 原论文 [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/abstract/document/726791) 中找到。

#### U-Net

!!! warning "To be completed"

#### FRCN

!!! warning "To be completed"

### 数据集

#### MNIST 手写数字数据集

MNIST 数据集 (Mixed National Institute of Standards and Technology database) 是美国国家标准与技术研究院收集整理的大型手写数字数据库，包含 60,000 个示例的训练集以及 10,000 个示例的测试集。

<div style="text-align:center;">
<img src="../graph/MNIST.jpeg" alt="How to Train a Model with MNIST dataset | by Abdullah Furkan Özbek | Medium" style="margin: 0 auto; zoom: 50%;"/>
</div>

一般给出的 MNIST 数据集下载链接为 http://yann.lecun.com/exdb/mnist/index.html，然而目前需要登录验证。建议使用 4.1.1 中 torchvision.datasets 的方法准备该数据集。 

#### [Carvana 数据集](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)

!!! warning "To be completed"

kaggle 上的一个语义分割竞赛数据集，完整的数据集比较大（20 GB+），不要求下载训练。

#### [NYU Depth 数据集](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

!!! warning "To be completed"

经典的密集标注的深度数据集，使用 RGBD 相机采样得到，不要求下载训练。

## 实验步骤

### LeNet-5 训练

#### 数据准备

建议利用 `torchvision` 提供的 `torchvision.datasets` 方法导入数据，`torchvision.datasets` 所提供的接口十分方便，之后你可以用 `torch.utils.data.DataLoader` 给你的模型加载数据。

幸运的是，本次实验需要用到的 `MNIST` 数据集可用 `torchvision.datasets` 导入，下面对一些你可能会用到的参数简单加以说明

!!! tip "请在清楚参数含义后调用它们"

```Python
# MNIST
torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```

一些重要的参数说明：

- `root`：数据集根目录，在 MNIST 中是 `processed/training.pt` 和 `processed/test.pt` 的主目录
- `train`：`True` 代表训练集，`False` 代表测试集
- `transform` 和 `target_transform`：分别是对图像和 label 的转换操作
- `download`：若为 `True` 则下载数据集并放到 `root` 所指定的目录中，否则直接尝试从 `root` 目录中读取

你可以在[这里](https://pytorch.org/vision/0.8/datasets.html)获取更加详细的说明

#### 模型编写

##### 网络结构

`PyTorch` 提供了许多种定义模型的方式，最常用的一种是将网络结构以类保存，你应当首先继承 [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)，并实现正向传播的 `forward` 函数，(为什么不用定义反向传播函数呢？因为你继承的 `nn.Module` 就是干这个事情的)。

下面为网络结构的一个 sample（但显然这样的网络并不能用于本次 Lab），本次实验中你**需要自定义你的网络结构**，以完成我们的分类任务：

```Python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # 利用参数初始化父类
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

当然，你需要实例化你的模型，可以直接对模型打印以查看结构

```Python
model = Model()
print(model)
```

网络结构编写中一个很大的难点在于每一步的 tensor shape 需要匹配，请仔细检查你的代码来确保此部分的正确性。

##### 损失函数

常见的损失函数都被定义在了 `torch.nn`中，你可以在训练过程开始前将其实例化，并在训练时调用，例如：

```Python
criterion = torch.nn.CrossEntropyLoss()
```

##### 正向传播

正向传播是指对神经网络沿着从输入层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）。
正向传播的过程在 `forward`中定义，对于模型实例，可以直接利用输入输出得到模型预测的结果。

```Python
y_pred = model(x)
```

##### 反向传播

反向传播（Backpropagation，BP）是“误差反向传播”的简称，是一种与最优化方法（如梯度下降法）结合使用的，用来训练人工神经网络的常见方法。该方法对网络中所有权重计算损失函数的梯度。这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数。

在计算过模型的loss之后，可以利用 `loss.backward()` 计算反向传播的梯度，梯度会被直接储存在 `requires_grad=True` 的节点中，不过此时节点的权重暂时不会更新，因此可以做到梯度的累加。

##### 优化器

常用的优化器都被定义在了 `torch.optim` 中，为了使用优化器，你需要构建一个 optimizer 对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。你需要给它一个包含了需要优化的参数（必须都是 Variable 对象）的iterable。然后，你可以设置optimizer的参数选项，比如学习率，权重衰减，例如：

```Python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

所有的 optimizer 都实现了 step() 方法，这个方法会更新所有的参数。或许你会在反向传播后用到它。

```Python
optimizer.step()
```

需要注意的是，在反向传播前，如果你不希望梯度累加，请使用下面的代码将梯度清零。

```Python
optimizer.zero_grad()
```

#### 训练过程

前文中已经定义了网络结构、损失函数、优化器，至此，一个较为完整的训练过程如下，需要注意的是，你的训练过程要不断从 `DataLoader` 中取出数据。

```Python
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### 测试过程
一般来说，神经网络会多次在训练集上进行训练，一次训练称之为一个 epoch。每个 epoch 结束后，我们会在测试集上进行测试，以评估模型的性能。在测试过程中，我们不需要计算梯度也不可以计算梯度（思考为什么），此时可以使用 `torch.no_grad` 来实现这一点。


```Python
with torch.no_grad():
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
```

#### Tips

- `nn.functional.relu`  （简记为 `F.relu` ）和 `nn.ReLU` 略有不同，区别在于前者作为一个函数调用，而后者作为一个层结构，必须添加到 `nn.Module` 容器中才能使用，两者实现的功能一样，在 `PyTorch` 中，`nn.X` 都有对应的函数版本 `F.X`。
- 除了利用继承 `nn.Module` 来建立网络，不推荐但可以使用 `nn.ModuleList`, `nn.ModuleDict`，推荐使用 `nn.Sequential`直接定义模型
- 你可以定义如下的 `device` 变量，以便你的模型在没有 GPU 环境下也可以测试：

    ```Python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)
    some_data = some_data.to(device)
    ```

- 你不必严格按照原版 LeNet-5 的网络结构来实现，包括超参数、优化器的选择不限，但是你需要保证你的网络结构是合理的，且能够完成我们的分类任务，**最终的测试集准确率需要达到 98% 以上**。（实际上原版 LeNet 可以轻松达到这个准确率，使用更加现代的结构和优化器，你可以达到更高的准确率）
- 不必过度关注准确率和 loss，评分将更关注有意义的探索过程记录而不是性能数值。


## 实验任务与要求

!!! warning "不允许直接使用各种深度学习开发工具已训练好的网络结构与参数"

!!! warning "参考文章、代码需在报告中列出，并体现出你的理解，否则一经查出视为抄袭"

1. LeNet-5：
      1. 使用 `PyTorch` 实现最基本的卷积神经网络 LeNet-5，并在 MNIST 数据集上进行训练
      2. 在测试集上进行测试，获得识别准确率，需要达到 98% 以上
      3. 由于 LeNet-5 太过经典、参考资料过多，代码的清晰程度和适当的原创注释也将是基本评分项
      4. (bonus) 对超参、优化器、网络结构等进行**有意义**的探索实验，将给予适当的 bonus。不鼓励无意义的内卷堆实验，评分时将酌情考虑。
2. U-Net 和 FRCN 的具体要求待定，将在之后更新
3. 你需要提交：
    1. 全部代码
    2. 实验报告，除了模板要求之外，还需要包含：
        1. 对于 LeNet-5，给出**模型的损失曲线、识别准确率曲线**等图表。可以利用 tensorboard 可视化训练过程并直接在其中截图，可以参考 [PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 的官方教程完成配置。
        2. 对于 LeNet-5，你需要写明测试集上的**识别正确率**
        3. U-Net 和 FRCN 的具体要求待定，将在之后更新
    3. 代码应单独打包为压缩文件，命名为 `学号-姓名-CVHW4` 的格式。实验报告应当单独上传附件，保证可以在网页直接打开实验报告进行预览，命名任意。

!!! warning "Deadline 之后交：按 80% 分数计算成绩"

## 参考资料

- [PyTorch 框架](https://pytorch.org/)
- [PyTorch Lightning 框架](https://www.pytorchlightning.ai/)
- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/index.html)（需验证）
- LeNet 原论文 [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/abstract/document/726791)
- [PyTorch 扩展](https://pytorch.org/docs/stable/notes/extending.html)
- [Dive into Deep Learning](https://d2l.ai/)
- [U-Net 的一种 Pytorch 实现](https://github.com/milesial/Pytorch-UNet)

## Acknowledgement

非常感谢 [chiakicage](https://github.com/chiakicage) 为另一门高质量课程编写的实验文档，让我能够在此基础上进行修改，让该实验得以在短期内发布。