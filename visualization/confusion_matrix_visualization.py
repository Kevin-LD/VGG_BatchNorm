import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from utils.eval import eval as custom_eval
from models.custom_cnn import CIFAR10ResidualNet


def calculate_confusion_matrix(preds, targets, num_classes=10):
    """
    仅使用 NumPy 计算混淆矩阵
    行 (Row): 真实标签 (True Targets)
    列 (Col): 预测标签 (Predictions)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        if 0 <= p < num_classes and 0 <= t < num_classes:
            cm[t, p] += 1
    return cm


def plot_and_save_confusion_matrices(cm, class_names, save_dir):
    """
    绘制并保存两张独立的混淆矩阵图表：原始计数图与百分比归一化图
    """
    num_classes = len(class_names)
    
    # 计算归一化矩阵 (按行求和)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.where(row_sums > 0, cm.astype('float') / row_sums, 0.0)

    # 1. 绘制并保存原始计数混淆矩阵 (Raw Counts)
    fig, ax1 = plt.subplots(figsize=(7.5, 6.5))
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title("Confusion Matrix (Raw Counts)", weight='bold', fontsize=12, pad=12)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 配置轴标签
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_yticks(np.arange(num_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(class_names, fontsize=9)
    ax1.set_xlabel("Predicted Label", weight='bold', fontsize=10, labelpad=5)
    ax1.set_ylabel("True Label", weight='bold', fontsize=10, labelpad=5)

    # 填充数值文本
    thresh1 = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            color = "white" if val > thresh1 else "black"
            ax1.text(j, i, f"{val}", ha="center", va="center", color=color, 
                     fontsize=8, weight='bold' if i == j else 'normal')
            
    plt.tight_layout()
    counts_save_path = os.path.join(save_dir, 'confusion_matrix_counts.png')
    plt.savefig(counts_save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 原始计数混淆矩阵已保存至: {counts_save_path}")
    plt.close()


    # 2. 绘制并保存归一化混淆矩阵 (Normalized Ratio)
    fig, ax2 = plt.subplots(figsize=(7.5, 6.5))
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='GnBu', vmin=0, vmax=1)
    ax2.set_title("Confusion Matrix (Normalized Ratio)", weight='bold', fontsize=12, pad=12)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 配置轴标签
    ax2.set_xticks(np.arange(num_classes))
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel("Predicted Label", weight='bold', fontsize=10, labelpad=5)
    ax2.set_ylabel("True Label", weight='bold', fontsize=10, labelpad=5)

    # 填充数值文本
    thresh2 = cm_normalized.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_normalized[i, j]
            color = "white" if val > thresh2 else "black"
            ax2.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, 
                     fontsize=8, weight='bold' if i == j else 'normal')
            
    plt.tight_layout()
    ratio_save_path = os.path.join(save_dir, 'confusion_matrix_ratio.png')
    plt.savefig(ratio_save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 百分比归一化混淆矩阵已保存至: {ratio_save_path}")
    plt.close()


def get_latest_model_checkpoint(base_dir='runs'):
    """自动扫描 runs 目录下最新的实验权重文件"""
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    
    checkpoint_path = os.path.join(latest_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
        
    all_files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.endswith('.pt')]
    return all_files[0] if all_files else None


def execute_confusion_matrix_pipeline(model_path=None):
    """
    混淆矩阵生成与可视化控制主流程
    """
    # 1. 自动定位实验目录与权重路径
    if model_path is None or model_path == "":
        model_path = get_latest_model_checkpoint('runs')
        if model_path is None:
            raise FileNotFoundError("[Error] 未在 runs/ 下检索到任何模型文件，请显式配置 model_path。")
        print(f"[*] 已自动捕获最新实验权重: {model_path}")
        
    target_experiment_dir = os.path.dirname(model_path)
    save_figures_dir = os.path.join(target_experiment_dir, 'figures')
    os.makedirs(save_figures_dir, exist_ok=True)
    
    # 2. 载入并解析 .pt 权重和元数据
    checkpoint = torch.load(model_path, map_location='cpu')
    metadata = checkpoint['metadata']
    config = metadata['config']
    
    # 3. 动态实例化模型
    print("[*] 正在解析超参数并构建网络计算图...")
    model = CIFAR10ResidualNet(
        base_channels=config['base_channels'],
        num_blocks=config.get('num_blocks', [2, 2, 2]),
        dropout_rate=config.get('dropout_rate', 0.0),
        activation_type=config['activation_type']
    )
    model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"[*] 模型已成功迁移至物理计算设备: {device}")

    # 4. 加载测试/验证数据集
    print("[*] 正在加载验证集数据流...")
    val_split = config.get('val_split', 0.1) if config.get('val_split', 0.0) > 0.0 else 0.1
    val_loader = get_cifar_loader(
        batch_size=config.get('batch_size', 128), 
        train=True, 
        val_split=val_split, 
        is_val=True, 
        shuffle=False,       
        num_workers=2
    )

    # 5. 调用前向传播以收集预测信号
    print("[*] 正在执行前向传播...")
    criterion = nn.CrossEntropyLoss()
    _, eval_acc, all_preds, all_targets = custom_eval(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        return_preds=True,
        desc="Collecting Predictions"
    )
    print(f"[*] 数据集前向评估完成。当前验证集准确率 (Accuracy): {eval_acc:.2%}")

    # 6. 计算混淆矩阵
    print("[*] 正在对齐预测标签序列并生成混淆矩阵...")
    cm = calculate_confusion_matrix(all_preds, all_targets, num_classes=10)

    # 7. 分别渲染并独立保存两幅可视化图表
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_and_save_confusion_matrices(cm=cm, class_names=cifar10_classes, save_dir=save_figures_dir)
    
    print("\n" + "=" * 60)
    print("     CONFUSION MATRIX VISUALIZATION COMPLETED SUCCESSFULLY     ")
    print(f" All assets are saved independently under: {save_figures_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    target_pt_path = "runs/sgd_ce_gelu_ch64_b333_20260605_121836_epoch_100/best_model.pt"
    execute_confusion_matrix_pipeline(model_path=target_pt_path)
