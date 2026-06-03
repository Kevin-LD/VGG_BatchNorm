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


class TransformedDataset(Dataset):
    """
    对 random_split 划分出的训练集和验证集应用完全不同的预处理与增强逻辑
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


def get_cifar_loader(root='./data/', batch_size=128, train=True, val_split=0.0, is_val=False, shuffle=True, num_workers=4, n_items=-1):
    """
    获取 CIFAR-10 的 DataLoader。
    支持标准数据增强（仅针对训练集生效）。
    支持通过 val_split 参数从训练集中划分出独立的验证集。
    """
    # 基础归一化配置
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    # 验证集与测试集的标准标准变换 (保持确定性，无数据增强)
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 策略 4(d): 针对训练集的数据增强流水线
    # 随机裁剪（四周补4像素再裁切回32x32） + 随机水平翻转
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # 载入基础数据集（注意：此处显式传入 transform=None，将变换递延至后续包装器处理）
    base_dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
    
    # 根据 train/val 的切分状态动态组装数据集
    if train:
        if val_split > 0.0:
            val_len = int(len(base_dataset) * val_split)
            train_len = len(base_dataset) - val_len
            
            # 使用固定的 manual_seed(42)
            train_subset, val_subset = torch.utils.data.random_split(
                base_dataset, 
                [train_len, val_len], 
                generator=torch.Generator().manual_seed(42)
            )
            
            # 根据需求指定返回动态增强的训练集，或者无增强的纯净验证集
            if is_val:
                dataset = TransformedDataset(val_subset, transform=val_transforms)
            else:
                dataset = TransformedDataset(train_subset, transform=train_transforms)
        else:
            # 未指定划分比例时，整个训练集全部应用增强
            dataset = TransformedDataset(base_dataset, transform=train_transforms)
    else:
        # 测试集严格执行无增强变换
        dataset = TransformedDataset(base_dataset, transform=val_transforms)

    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


if __name__ == '__main__':
    # CIFAR-10 类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 验证划分逻辑与增强隔离是否正确
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
    collected_samples = {i: [] for i in range(10)}
    
    # 遍历切分出的训练集进行数据收集与可视化展示
    for X, y in train_loader:
        if all(len(images) == num_per_class for images in collected_samples.values()):
            break
            
        for i in range(X.shape[0]):
            label = y[i].item()
            
            if len(collected_samples[label]) < num_per_class:
                # 转换维度 (C, H, W) -> (H, W, C)
                img = X[i].permute(1, 2, 0).numpy()
                # 反归一化
                img = img * 0.5 + 0.5
                img = np.clip(img, 0, 1)
                
                collected_samples[label].append(img)
                
        if all(len(images) == num_per_class for images in collected_samples.values()):
            break

    # 创建一个 10 行、num_per_class 列的画布
    fig, axes = plt.subplots(nrows=10, ncols=num_per_class, figsize=(num_per_class * 2, 15))
    
    for class_idx in range(10):
        for sample_idx in range(num_per_class):
            ax = axes[class_idx, sample_idx]
            
            # 渲染图像（此处可以看到因为数据增强产生的黑边、平移或水平翻转效果）
            ax.imshow(collected_samples[class_idx][sample_idx])
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if sample_idx == 0:
                ax.set_ylabel(classes[class_idx], rotation=0, ha='right', va='center', fontsize=12)

    plt.tight_layout()
    
    # 确保保存路径存在
    output_dir = './figures'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'cifar10_augmented_samples.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"数据增强后的可视化结果已成功保存至: {save_path}")
