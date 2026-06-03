"""
Data loaders
"""
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root='./data/', batch_size=128, train=True, val_split=0.0, is_val=False, shuffle=True, num_workers=4, n_items=-1):
    """
    获取 CIFAR-10 的 DataLoader。
    支持通过 val_split 参数从训练集中划分出独立的验证集。
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 载入基础数据集
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    
    # 如果是训练模式且指定了划分比例，则从训练集中切分出验证集
    if train and val_split > 0.0:
        val_len = int(len(dataset) * val_split)
        train_len = len(dataset) - val_len
        
        # 核心：使用固定的 manual_seed(42) 确保多次调用此函数时，训练/验证的划分逻辑完全一致
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_len, val_len], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # 根据需求指定返回训练子集或验证子集
        dataset = val_dataset if is_val else train_dataset

    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


if __name__ == '__main__':
    # CIFAR-10 类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 验证划分逻辑是否正确
    # 从 50000 张训练集中划分出 10% (5000张) 作为验证集，剩余 45000 张作为训练集
    train_loader = get_cifar_loader(batch_size=128, train=True, val_split=0.1, is_val=False, shuffle=True)
    val_loader = get_cifar_loader(batch_size=128, train=True, val_split=0.1, is_val=True, shuffle=False)
    test_loader = get_cifar_loader(batch_size=128, train=False, shuffle=False)
    
    print("----- Dataloader Split Verification -----")
    print(f"Train set batches: {len(train_loader)} (Total images: {len(train_loader.dataset)})")
    print(f"Val set batches:   {len(val_loader)} (Total images: {len(val_loader.dataset)})")
    print(f"Test set batches:  {len(test_loader)} (Total images: {len(test_loader.dataset)})")
    print("-" * 42)

    # 每个类别需要固定展示的图片数量
    num_per_class = 10

    # 初始化一个字典，用于存放每个类别收集到的图像
    # 键为类别索引 (0-9)，值为图像矩阵的列表
    collected_samples = {i: [] for i in range(10)}
    
    # 遍历零时切分出的训练集进行数据收集与可视化展示
    for X, y in train_loader:
        # 检查是否所有类别都已集齐所需数量
        if all(len(images) == num_per_class for images in collected_samples.values()):
            break
            
        for i in range(X.shape[0]):
            label = y[i].item()
            
            # 如果当前类别还没集齐，则进行处理并加入
            if len(collected_samples[label]) < num_per_class:
                # 转换维度 (C, H, W) -> (H, W, C)
                img = X[i].permute(1, 2, 0).numpy()
                # 反归一化
                img = img * 0.5 + 0.5
                img = np.clip(img, 0, 1)
                
                collected_samples[label].append(img)
                
        # 再次检查，以便在 batch 内部集齐时能及时跳出外层循环
        if all(len(images) == num_per_class for images in collected_samples.values()):
            break

    # 创建一个 10 行、num_per_class 列的画布
    # 每行代表一个类别，每列代表该类别的一张样本图
    fig, axes = plt.subplots(nrows=10, ncols=num_per_class, figsize=(num_per_class * 2, 15))
    
    for class_idx in range(10):
        for sample_idx in range(num_per_class):
            # 获取对应的子图对象
            ax = axes[class_idx, sample_idx]
            
            # 渲染图像
            ax.imshow(collected_samples[class_idx][sample_idx])
            
            # 清除坐标轴刻度
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 仅在每一行的第一列左侧显示类别名称作为行标签
            if sample_idx == 0:
                ax.set_ylabel(classes[class_idx], rotation=0, ha='right', va='center', fontsize=12)

    # 调整布局，防止标签重叠
    plt.tight_layout()
    
    # 确保保存路径存在
    output_dir = './figures'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'cifar10_samples.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"数据集可视化结果已成功保存至: {save_path}")
