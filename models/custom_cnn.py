"""
Custom Network Architectures for CIFAR-10 Classification
"""
import torch
import torch.nn as nn
import numpy as np


def get_number_of_parameters(model):
    """计算模型总参数量"""
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n


def get_activation(activation_type="relu"):
    """
    策略 4(c): 动态选择不同的激活函数
    """
    activation_type = activation_type.lower()
    if activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif activation_type == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


class ResidualBlock(nn.Module):
    """
    组件 3(c): 残差连接块
    包含 2D 卷积、Batch Normalization、激活函数及捷径分支
    """
    def __init__(self, in_channels, out_channels, stride=1, activation_type="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 组件 3(a): Batch-Norm
        self.act1 = get_activation(activation_type)  # 组件 2(d): 激活函数
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = get_activation(activation_type)
        
        # 如果输入输出维度不一致，使用 1x1 卷积调整残差分支的维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差相加
        out = self.act2(out)
        return out


class CIFAR10ResidualNet(nn.Module):
    """
    自定义 CIFAR-10 分类网络
    满足要求:
    - 2(a) 全连接层
    - 2(b) 2D 卷积层
    - 2(c) 2D 池化层
    - 2(d) 激活函数
    - 3(a) Batch-Norm 层
    - 3(b) Dropout
    - 3(c) 残差连接
    - 4(a) 参数化配置滤波器/神经元数量
    """
    def __init__(self, base_channels=32, num_classes=10, dropout_rate=0.3, activation_type="relu"):
        super().__init__()
        
        # 策略 4(a): 通过 base_channels 缩放整个网络的滤波器/通道数量
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # 阶段 1: 初始特征提取
        self.prep = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            get_activation(activation_type)
        )

        # 阶段 2: 包含残差块与多尺度池化
        # 组件 2(b): 2D 卷积隐含在残差块内部
        self.layer1 = ResidualBlock(c1, c1, stride=1, activation_type=activation_type)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 组件 2(c): 2D 池化

        self.layer2 = ResidualBlock(c1, c2, stride=1, activation_type=activation_type)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = ResidualBlock(c2, c3, stride=1, activation_type=activation_type)
        # 采用自适应平均池化，将特征图大小固定为 1x1，提高对不同输入尺寸的鲁棒性并减少全连接层参数量
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        # 阶段 3: 分类器
        # 组件 3(b): Dropout 层，用于正则化防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 组件 2(a): 全连接层 / 线性层
        # 内部隐含策略 4(a): 隐层神经元数量与全局通道数 c3 绑定
        self.fc1 = nn.Linear(c3, c3 // 2)
        self.act_fc = get_activation(activation_type)
        self.fc2 = nn.Linear(c3 // 2, num_classes)

    def forward(self, x):
        # 提取特征图（暴露中间层便于后续 Task 1.1.6 要求的滤波器与特征可视化）
        out = self.prep(x)
        out = self.pool1(self.layer1(out))
        out = self.pool2(self.layer2(out))
        out = self.pool3(self.layer3(out))
        
        # 展平张量用于全连接层
        out = torch.flatten(out, 1)
        
        # 分类逻辑
        out = self.dropout(out)
        out = self.act_fc(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    # 测试代码：验证网络前向传播维度及参数量统计
    # 策略 4(a) & 4(c) 示例配置
    test_model = CIFAR10ResidualNet(base_channels=32, activation_type="gelu", dropout_rate=0.2)
    
    # 模拟一个标准 CIFAR-10 batch 的输入: [Batch_size, Channels, Height, Width]
    mock_input = torch.randn(4, 3, 32, 32)
    mock_output = test_model(mock_input)
    
    print("----- Model Verification -----")
    print(f"Input shape:  {mock_input.shape}")
    print(f"Output shape: {mock_output.shape}")  # 预期输出: torch.Size([4, 10])
    print(f"Total Parameters: {get_number_of_parameters(test_model):,}")
