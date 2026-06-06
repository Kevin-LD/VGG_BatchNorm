import os
import sys
import random

# 动态获取当前脚本的绝对路径，并将父目录（项目根目录）加入系统路径
# 确保在任意工作目录下直接执行本脚本时，均能正确导入 data、utils 和 models 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 导入自定义的数据加载器、评估函数与网络结构
from data.loaders import get_cifar_loader
from utils.eval import eval as custom_eval
from models.custom_cnn import CIFAR10ResidualNet


def collect_misclassified_examples_randomly(model, loader, device, max_errors=12, seed=None):
    """
    遍历完整的数据加载器，收集全量预测错误的样本，并从中随机抽样指定数量。
    支持设置 seed 以保证实验的可重复性。
    """
    if seed is not None:
        random.seed(seed)
        
    model.eval()
    all_misclassified = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_dev = X_batch.to(device)
            outputs = model(X_dev)
            _, predicted = torch.max(outputs, 1)
            
            # 转换为 NumPy 进行逐元素比对
            preds_np = predicted.cpu().numpy()
            targets_np = y_batch.numpy()
            
            # 筛选出当前 Batch 中错例的索引
            error_indices = np.where(preds_np != targets_np)[0]
            
            for idx in error_indices:
                img_tensor = X_batch[idx].numpy()
                pred_label = preds_np[idx]
                true_label = targets_np[idx]
                
                all_misclassified.append({
                    'image': img_tensor,
                    'pred': pred_label,
                    'true': true_label
                })
                
    total_errors = len(all_misclassified)
    print(f"[*] 全量前向传播完成。整个数据集中共检测到错例数: {total_errors} 个。")
    
    if total_errors == 0:
        return []
        
    # 如果实际错例总数小于或等于需要的抽样数，则直接返回全部错例
    if total_errors <= max_errors:
        return all_misclassified
        
    # 从全量错例中进行无放回随机抽样
    print(f"[*] 正在从 {total_errors} 个错例中随机抽取 {max_errors} 个样本进行可视化...")
    return random.sample(all_misclassified, max_errors)


def plot_and_save_error_grid(errors, class_names, save_path, rows=3, cols=4):
    """
    将收集到的随机错例以网格矩阵（Grid）的形式进行排版可视化
    """
    total_plots = rows * cols
    actual_errors = len(errors)
    
    if actual_errors == 0:
        print("[Warning] 未检索到任何错例，跳过绘图。")
        return

    # 如果实际错例数少于预设网格大小，动态缩小网格
    display_count = min(total_plots, actual_errors)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3.0))
    axes = axes.flatten()

    # CIFAR-10 数据集的反归一化参数配置
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    for i in range(total_plots):
        ax = axes[i]
        if i < display_count:
            err_item = errors[i]
            
            # 维度转换与反归一化处理: [3, 32, 32] -> [32, 32, 3]
            raw_img = err_item['image'].transpose(1, 2, 0)
            raw_img = (raw_img * std) + mean
            raw_img = np.clip(raw_img, 0.0, 1.0)
            
            # 渲染图像（使用 lanczos 算法进行平滑抗锯齿缩放）
            ax.imshow(raw_img, interpolation='lanczos')
            ax.axis('off')
            
            true_str = class_names[err_item['true']]
            pred_str = class_names[err_item['pred']]
            
            # 设置红黑对比标题
            ax.set_title(f"True: {true_str}\nPred: {pred_str}", 
                         fontsize=10, 
                         weight='bold',
                         color='darkred', 
                         pad=6)
        else:
            ax.axis('off')

    plt.suptitle("Randomized Error Analysis: Misclassified Examples", weight='bold', fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 随机错例可视化分析网格图已成功保存至: {save_path}")
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


def execute_error_analysis_pipeline(model_path=None, grid_rows=3, grid_cols=4, random_seed=42):
    """
    错例收集与随机抽样可视化核心控制流
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

    # 4. 加载确定性的验证/测试数据集
    print("[*] 正在加载验证集数据流...")
    val_split = config.get('val_split', 0.1) if config.get('val_split', 0.0) > 0.0 else 0.1
    val_loader = get_cifar_loader(
        batch_size=config.get('batch_size', 128), 
        train=True, 
        val_split=val_split, 
        is_val=True, 
        shuffle=False,       # 错例位置检索严禁在 loader 层打乱序列
        num_workers=2
    )

    # 5. 调用标准 eval 函数获取当前全局准确率
    print("[*] 正在基准化整体评估指标...")
    criterion = nn.CrossEntropyLoss()
    _, eval_acc = custom_eval(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        return_preds=False,
        desc="Baselining Accuracy"
    )
    print(f"[*] 基准校验完成。当前验证集准确率 (Accuracy): {eval_acc:.2%}")

    # 6. 执行全网错例拦截并进行随机抽样
    max_to_show = grid_rows * grid_cols
    error_samples = collect_misclassified_examples_randomly(
        model=model, 
        loader=val_loader, 
        device=device, 
        max_errors=max_to_show,
        seed=random_seed
    )

    # 7. 排版并留存高清可视化矩阵大图
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    save_fig_path = os.path.join(save_figures_dir, 'misclassified_examples_random.png')
    
    plot_and_save_error_grid(
        errors=error_samples, 
        class_names=cifar10_classes, 
        save_path=save_fig_path,
        rows=grid_rows,
        cols=grid_cols
    )
    
    print("\n" + "=" * 60)
    print("     RANDOMIZED ERROR VISUALIZATION COMPLETED SUCCESSFULLY     ")
    print(f" Randomized error asset is saved safely under: {save_figures_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # target_pt_path: 指定 .pt 路径。若为 None 则自动检索 runs/ 下最新权重。
    # rows, cols: 错例展示矩阵的行数与列数（相乘即为展示错例的总数）。
    # random_seed: 随机数种子。固定种子可以确保多次运行脚本时抽到相同的错例，
    #              如果希望每次运行完全随机，可以将其设为 None。
    target_pt_path = "runs/sgd_ce_gelu_ch64_b333_20260605_121836_epoch_100/best_model.pt"
    rows = 3
    cols = 4
    seed = 42
    
    execute_error_analysis_pipeline(
        model_path=target_pt_path,
        grid_rows=rows,
        grid_cols=cols,
        random_seed=seed
    )
