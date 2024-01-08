<link rel="stylesheet" href="../css/counter.css" />

# HW5: Learning CNN++

## 实验简介

- [Kaggle](https://zh.wikipedia.org/wiki/Kaggle)：谷歌旗下的数据科学竞赛平台，有开放的比赛、数据集、模型、讨论等，用户可以参与比赛打榜、获取数据集练习、分享自己模型与经验
- Jupyter Notebook：非营利组织 [Jupyter](https://zh.wikipedia.org/wiki/Jupyter) 的产品，是一个开源的基于 Web 的交互式计算环境，允许用户在 .ipynb 文档中分享 markdown 文本、代码以及代码执行结果

在上次实验学习 CNN 的基础上，本次实验我们将使用 CNN 完成 [Kaggle](https://zh.wikipedia.org/wiki/Kaggle) 公开数据集 [Pokemon Image Dataset](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/data) 上的精灵宝可梦类型预测任务。

## 实验环境

要求使用 python + pytorch 完成实验，不允许使用参考仓库中的 tensorflow 代码。

## 实验基础知识介绍

### CNN 结构简述

上次实验学习的 LeNet 奠定了一种 CNN 的经典结构：卷积层与池化层交替出现，后接一系列全连接层，最终得到输出。这种经典结构也在 AlexNet 中得到沿用，并且在 2012 年一鸣惊人。你将会注意到，本次实验的参考案例也会用到这种结构。

后来，CNN 的结构又有如下的一些发展：

- 由于全连接层可以看成一种卷积的观点出现，**全卷积 (full convolution)** 网络出现
- 随着 CNN 可视化研究的进展，CNN 中间层的输出被认为是**不同尺度的特征提取**
- 上次实验所提到的 U-Net，则是将不同尺度的信息通过一种 skip 与 concat 的方式组合利用，在图像分割上达到了很好的效果
- 随着 VGG、ResNet 这些在当时被认为参数量足够大的模型出现，在预训练模型的参数基础上更改网络结构进行**微调 (finetune)** 的技术也逐渐成熟起来
- ......

随着 Transformer 从机器翻译领域开始掀起的一场变革，Vision Transformer (ViT) 把 Attention 也带到了视觉领域。在本学期课程中新增内容中也包含 Attention，你也可以尝试借鉴 Attention 在视觉领域的应用，在你的模型中加入 Attention。

### 数据集简介

[Pokemon Image Dataset](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/data) 是 [Kaggle](https://zh.wikipedia.org/wiki/Kaggle) 上的一个公开数据集，包含了 1 代到 7 代所有的宝可梦 (Pokemon)，每种宝可梦的类型信息（Type1 & Type2）都被包含在 `pokemon.csv` 中。

由于 Type2 对于某些宝可梦来说是空的，因此需要必要的数据预处理。为了减少大家无关 CNN 的工作量，提供了一个案例可以参考学习。

### 数据增广

数据增广 (data augmentation) 是在当前有限的图像数据集的基础上增加数据量的一种方法，常见的数据增广方法是对原始的图像进行仿射变换而标签不变，另外像取 patch 也是一种常见的数据增广方式。更多数据增广的方法有待探索。

可以参考 [Dive into Deep Learning - Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html) 学习数据增广的使用。

## 实验设置

### 参考案例

给出一个可参考的案例 [pokemon-types](https://github.com/rshnn/pokemon-types)，注意该 github 仓库没有给出 LICENSE，默认一切权利由作者所有，因此大家注意进行借鉴学习。提交的内容不能是 .ipynb 文件。

可以参考该案例中的数据预处理、输入输出适配，减少与 CNN 无关的工作量。该仓库中给出了用 tensorflow.keras 实现的一种 CNN 结构，可以作为 baseline，但是必须自己使用 PyTorch 实现。

另外 Kaggle 平台上的讨论也有很多可以参考学习的内容，大家可以自行探索。

### 训练集-测试集划分

对于训练集和测试集划分，使用统一的 [train.csv](csv/train.csv) 和 [test.csv](csv/test.csv)，分别包含 687 只和 122 只宝可梦。

不允许使用别的训练集、测试集设置，不必引入交叉验证集。

### 测试集泄露

- 要求的训练集含 687 只宝可梦，测试集含 122 只宝可梦
- 测试集泄露指使用训练集和测试集全部数据（共 809 只宝可梦）进行训练
- 而正确的操作是不应该泄露测试集，只根据训练集 687 只宝可梦进行训练

如果测试集泄露，理论上可以让模型“记住”测试集的数据，从而达到神乎其神的 100% 测试准确率，而这显然是不对的。

## 实验任务与要求

!!! warning "不允许直接使用各种深度学习开发工具已训练好的网络结构与参数"
    扩展要求中允许利用预训练模型进行微调。

!!! warning "参考文章、代码需在报告中列出，并体现出你的理解，否则一经查出视为抄袭"

### 实验要求

**基本要求**包括：

1. 使用 PyTorch 自己设计 CNN，**自己训练**完成 [Pokemon Image Dataset](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/data) 上的精灵宝可梦类型预测任务
> 可以自己设计 CNN，可以参考案例中的 CNN，也可以修改 LeNet 等结构，但是不允许使用预训练参数，必须从头训练
2. 应用**数据增广**方法
3. **调整超参数**进行对比实验，对测试集上的准确率等实验结果进行**比较分析**
4. 要求在测试集泄露和不泄露两种情况下进行训练，在测试集上进行测试对比

> 点击下载要求的训练集、测试集划分配置：[train.csv](csv/train.csv) 和 [test.csv](csv/test.csv)

**扩展要求**为尝试修改网络结构甚至采用新的架构应用于该任务，进行实验与比较。实验基础知识部分提供了一些可以参考的思路。

### 提交要求

**截止时间：2024 年 1 月 24 日上午**，详见[学在浙大](https://courses.zju.edu.cn)

!!! warning "大家特别注意：ddl 之后马上要批改并提交总成绩，所以若 ddl 前没能提交该作业将按零分计算"

你需要提交：

1. 全部代码
2. 实验报告，除了模板要求之外，对比实验要求有分析图表，并写明测试集上的**识别正确率**
3. 代码应单独打包为压缩文件，命名为 `学号-姓名-CVHW5` 的格式。实验报告应当单独上传附件，保证可以在网页直接打开实验报告进行预览，命名任意。

## 参考资料

- [Pokemon Image Dataset](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/data)
- 参考案例仓库 [pokemon-types](https://github.com/rshnn/pokemon-types)
- [Dive into Deep Learning - Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html)