# Residual CNN and Batch Normalization Analysis

## 项目简介

本项目包含两个部分：

### Task 1: CIFAR-10 图像分类

设计并实现了一个基于残差结构（Residual Block）的卷积神经网络，并在 CIFAR-10 数据集上进行了超参数搜索与消融实验。最终模型取得 **93.37% Test Accuracy**。

### Task 2: Batch Normalization 优化机制研究

基于 VGG-A 网络研究 Batch Normalization 对深度神经网络优化过程的影响。

实验复现了论文《Batch Normalization Helps Optimization》中的核心结论，并从以下角度分析 BN 的作用：

* Loss Landscape
* Gradient Predictiveness
* Effective β-Smoothness

结果表明 Batch Normalization 能够显著平滑优化景观，提高梯度稳定性，从而改善模型训练效率与最终性能。

---

## 环境配置

实验环境：

* Ubuntu 22.04.5 LTS (WSL)
* Python 3.12.13

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 数据准备

首次运行训练脚本时会自动下载 CIFAR-10 数据集。

也可以手动下载数据：

数据集地址：[Google Drive Link](https://drive.google.com/drive/folders/11pCEekqYjdVTSe6RI-Tni7DoN8hNSLGx?usp=sharing)

---

## 运行方式

### Task 1：Residual CNN

#### 模型训练

```bash
python task1_train.py
```

如有需要，可在脚本中修改训练配置。

#### 模型测试

```bash
python task1_eval.py
```

运行前请在脚本中填写待测试模型权重路径。

---

### Task 2：Batch Normalization Analysis

#### Loss Landscape（课程作业简化版本）

```bash
python VGG_Loss_Landscape.py
```

按照作业要求，通过不同学习率训练多个模型，并绘制 Loss Variation 曲线。

运行后将生成 Loss Landscape 可视化结果。

#### Santurkar et al. 原始实验复现

```bash
python VGG_Loss_Landscape_Original.py
```

运行后将生成：

* Loss Landscape
* Gradient Predictiveness
* Effective β-Smoothness

三项实验结果图。

#### 模型测试

```bash
python task2_eval.py
```

运行前请在脚本中填写待测试模型权重路径。

---

## Benchmark

使用以下脚本统计 Task 1 与 Task 2 中的模型规模与运行效率：

```bash
python benchmark.py
```

输出内容包括：

* 模型参数量
* FLOPs
* 训练速度

---

## 可视化工具

项目中的所有可视化脚本均位于：

```text
visualization/
```

根据需求在脚本中修改：

* 实验目录路径
* 模型权重路径

随后直接运行对应脚本即可生成可视化结果。

---

## 模型权重

模型权重下载地址：[Google Drive Link](https://drive.google.com/drive/folders/1bi55b1brWIilnt8N03-trcwOrMYxJ2VF?usp=sharing)
