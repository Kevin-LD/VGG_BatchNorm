import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_first_layer_weights(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到指定的模型 checkpoint 文件: {model_path}")
        
    # 转换为 CPU 张量进行分析
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 解析出真实的 state_dict 字典
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    else:
        # 如果保存的是整个模型实例而非 state_dict
        state_dict = checkpoint.state_dict()
        
    # 自动检索第一个符合 [out_channels, in_channels, k_h, k_w] 结构的 4D 权重张量
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) == 4:
            # 过滤失真项或非权重项（如某些架构中的 padding 辅助张量）
            if 'weight' in key.lower() and not 'bn' in key.lower():
                weights_np = tensor.detach().numpy()
                print(f"[*] 成功检索到第一层卷积权重: '{key}' | 形状: {weights_np.shape}")
                return weights_np, key
                
    raise ValueError(f"未能从文件 {model_path} 的权重字典中解析出任何 4D 卷积层张量。")


def visualize_channels_diverging(weights, save_dir, layer_name):
    out_channels, in_channels, k_h, k_w = weights.shape
    
    # 动态计算子图网格布局
    cols = int(np.ceil(np.sqrt(out_channels)))
    rows = int(np.ceil(out_channels / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()
    
    # 计算全局绝对值最大值，用于零中心对称对齐 (保证 0 值精确映射为白色)
    max_val = np.max(np.abs(weights[:, 0, :, :]))
    
    for i in range(out_channels):
        ax = axes[i]
        # 提取当前卷积核在第 0 个通道的空间权重
        kernel = weights[i, 0, :, :]
        
        im = ax.imshow(kernel, cmap='RdBu_r', vmin=-max_val, vmax=max_val, interpolation='nearest')
        
        ax.set_title(f"F{i}", fontsize=8, pad=2)
        ax.axis('off')
        
    # 隐藏多余的空白子图占位符
    for j in range(out_channels, len(axes)):
        axes[j].set_visible(False)
        
    # 在右侧增加全局统一的 Colorbar 刻度条
    fig.subplots_adjust(right=0.88, top=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    fig.suptitle(f"First Layer Convolutional Kernels Visualization\nLayer: {layer_name} (Channel 0)", weight='bold', y=0.96)
    
    save_path = os.path.join(save_dir, 'weight_kernels_diverging.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 空间发散特征图已成功保存至: {save_path}")
    plt.close()


def visualize_rgb_kernels(weights, save_dir, layer_name):
    """
    将 3 通道（RGB）卷积核进行 Min-Max 归一化，并渲染为彩色图像。
    用于分析模型对颜色互补色、彩色边缘的捕获能力。
    """
    out_channels, in_channels, k_h, k_w = weights.shape
    if in_channels != 3:
        # 如果输入不是 3 通道（如灰度图），跳过彩色渲染
        return
        
    cols = int(np.ceil(np.sqrt(out_channels)))
    rows = int(np.ceil(out_channels / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()
    
    for i in range(out_channels):
        ax = axes[i]
        kernel = weights[i, :, :, :]  # 形状为 [3, k_h, k_w]
        # 调换维度以符合 matplotlib [k_h, k_w, 3] 的 RGB 要求
        kernel = np.transpose(kernel, (1, 2, 0))
        
        # 进行严格的通道域 Min-Max 归一化，映射至 [0, 1] 空间用于规范显示
        k_min, k_max = kernel.min(), kernel.max()
        if k_max > k_min:
            kernel_normalized = (kernel - k_min) / (k_max - k_min)
        else:
            kernel_normalized = kernel - k_min
            
        ax.imshow(kernel_normalized, interpolation='nearest')
        ax.set_title(f"F{i}", fontsize=8, pad=2)
        ax.axis('off')
        
    for j in range(out_channels, len(axes)):
        axes[j].set_visible(False)
        
    fig.suptitle(f"First Layer Convolutional Kernels RGB Render\nLayer: {layer_name}", weight='bold', y=0.95)
    
    save_path = os.path.join(save_dir, 'weight_kernels_rgb.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 卷积核 RGB 彩色渲染图已成功保存至: {save_path}")
    plt.close()


def get_latest_model_checkpoint(base_dir='runs'):
    """自动扫描 runs 目录下时间戳最新的实验目录，并检索其中的最优模型权重"""
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('search_')]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    
    # 检索该线下最可能存在的模型权重命名
    possible_names = ['best_model.pth', 'model.pth', 'checkpoint.pth']
    for name in possible_names:
        full_path = os.path.join(latest_dir, name)
        if os.path.exists(full_path):
            return full_path
            
    # 如果没找到标准命名，模糊匹配目录下任意 .pth 或 .pt 文件
    all_files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.endswith(('.pth', '.pt'))]
    return all_files[0] if all_files else None


def execute_weight_visualization(model_path=None):
    # 1. 自动定位目标模型路径与实验目录
    if model_path is None or model_path == "":
        model_path = get_latest_model_checkpoint('runs')
        if model_path is None:
            raise FileNotFoundError("[Error] 未在 runs/ 目录下检索到任何模型文件，请显式指定 model_path。")
        print(f"[*] 未指定输入模型，已自动捕获最新实验权重: {model_path}")
        
    target_experiment_dir = os.path.dirname(model_path)
    
    # 2. 强制在当前模型所在的实验目录下建立 figures 文件夹
    save_figures_dir = os.path.join(target_experiment_dir, 'figures')
    os.makedirs(save_figures_dir, exist_ok=True)
    
    print("-" * 60)
    print(f"  WEIGHT VISUALIZATION ENGINE LAUNCHED")
    print(f"  Target Model: {model_path}")
    print(f"  Output Folder: {save_figures_dir}")
    print("-" * 60)
    
    try:
        # 3. 提取权重并执行可视化操纵
        weights, layer_name = load_first_layer_weights(model_path)
        
        # 生成标准发散特征图
        visualize_channels_diverging(weights, save_figures_dir, layer_name)
        
        # 尝试生成 RGB 彩色渲染图 (仅在输入通道为 3 时生效)
        if weights.shape[1] == 3:
            visualize_rgb_kernels(weights, save_figures_dir, layer_name)
            
        print("\n" + "=" * 60)
        print("         WEIGHT VISUALIZATION COMPLETED SUCCESSFULLY        ")
        print(f" All figures are saved safely under: {save_figures_dir}")
        print("=" * 60 + "\n")
        
    except Exception as error:
        print(f"[Fatal Error] 权重分析中断，错误原因: {error}")
        raise error


if __name__ == "__main__":
    # 可以直接在此处填入具体想分析的模型文件路径，例如：
    # target_model = "runs/search_20260603_223602/best_model.pt"
    # 若保持为 None，系统会自动寻找 runs/ 下最新生成的实验目录并提取模型权重。
    target_model = "runs/sgd_ce_gelu_ch64_b333_20260605_121836_epoch_100/best_model.pt"
    
    execute_weight_visualization(model_path=target_model)