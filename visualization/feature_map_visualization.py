import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders import get_cifar_loader
from models.custom_cnn import CIFAR10ResidualNet


class MultiLayerActivationHook:
    """
    多层前向钩子管理器，用于同时跟踪和拦截网络中多个指定层的输出。
    """
    def __init__(self):
        self.all_activated_features = {}

    def get_hook_fn(self, layer_name):
        def hook(module, module_in, module_out):
            self.all_activated_features[layer_name] = module_out.detach().cpu().numpy()
        return hook


def get_real_cifar_sample(val_split=0.1):
    """
    利用项目现有的数据加载器，从验证集中随机抽取一张真实的 CIFAR-10 图像。
    根据传入的 loader 配置，使用相应的均值和标准差进行反归一化。
    """
    val_loader = get_cifar_loader(
        batch_size=1, 
        train=True, 
        val_split=val_split, 
        is_val=True, 
        shuffle=True, 
        num_workers=0  
    )
    
    iterator = iter(val_loader)
    input_tensor, label = next(iterator)
    
    # 反归一化
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    raw_img = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    raw_img = (raw_img * std) + mean
    raw_img = np.clip(raw_img, 0.0, 1.0)
    
    return input_tensor, raw_img, label.item()


def create_structured_synthetic_input(in_channels=3, height=32, width=32):
    """
    结构化几何合成输入张量（备用方案）。
    """
    img = np.ones((height, width), dtype=np.float32) * 0.1
    cy, cx = height // 2, width // 2
    r = min(height, width) // 4
    img[cy-r:cy+r, cx-r:cx+r] = 0.8
    img[cy-1:cy+2, :] = 0.9
    img[:, cx-1:cx+2] = 0.9
    
    img_stacked = np.stack([img] * in_channels, axis=0)
    input_tensor = torch.from_numpy(img_stacked).unsqueeze(0)
    return input_tensor, img


def plot_network_feature_hierarchy(all_features, save_dir, filename, channels_to_show=4):
    """
    分层排版抽样后的网络特征图。
    矩阵布局：每一行代表网络的一个精选卷积层，每一列代表该层的前 K 个特征通道。
    """
    layer_names = list(all_features.keys())
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("[Warning] 未截获到任何层的特征图数据，跳过绘图。")
        return

    # 画布尺寸自适应调整
    fig, axes = plt.subplots(num_layers, channels_to_show, 
                             figsize=(channels_to_show * 2.0, num_layers * 1.8),
                             squeeze=False)
    
    for row_idx, layer_name in enumerate(layer_names):
        f_map = np.squeeze(all_features[layer_name], axis=0)
        out_channels, h, w = f_map.shape
        actual_cols = min(out_channels, channels_to_show)
        
        for col_idx in range(channels_to_show):
            ax = axes[row_idx, col_idx]
            
            if col_idx < actual_cols:
                channel_map = f_map[col_idx, :, :]
                ax.imshow(channel_map, cmap='viridis', interpolation='nearest')
                
                if col_idx == 0:
                    ax.set_ylabel(layer_name, fontsize=8, rotation=0, labelpad=45, ha='right', weight='bold')
                ax.set_title(f"Ch {col_idx}", fontsize=8, pad=2)
            else:
                ax.axis('off')
                
            ax.set_xticks([])
            ax.set_yticks([])
            
    fig.suptitle(f"Sampleed Network Feature Activation Hierarchy\nRows: Selected Key Layers | Columns: First {channels_to_show} Channels", 
                 weight='bold', y=0.99, fontsize=11)
    
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    fig.subplots_adjust(top=0.94 - (0.01 * (num_layers // 3)))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 网络层级特征图已成功保存至: {save_path}")
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


def execute_network_visualization(model_path=None, max_layers_to_show=6, max_channels_per_layer=4, use_real_image=True):
    """
    全网络特征层级精简可视化控制流
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
    
    # 2. 载入 .pt 文件并解析出元数据字典
    checkpoint = torch.load(model_path, map_location='cpu')
    metadata = checkpoint['metadata']
    config = metadata['config']
    
    # 3. 动态实例化模型
    model = CIFAR10ResidualNet(
        base_channels=config['base_channels'],
        num_blocks=config.get('num_blocks', [2, 2, 2]),
        dropout_rate=config.get('dropout_rate', 0.0),
        activation_type=config['activation_type']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 4. 【核心改动】扫描并对全网卷积层进行拓扑均匀抽样
    all_conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            all_conv_layers.append((name, module))
            
    total_conv_count = len(all_conv_layers)
    
    if total_conv_count <= max_layers_to_show:
        sampled_conv_layers = all_conv_layers
    else:
        # 使用 linspace 计算等间距索引，确保绝对包含输入首层与输出尾层
        indices = np.linspace(0, total_conv_count - 1, num=max_layers_to_show, dtype=int)
        indices = sorted(list(set(indices))) # 去重并保持顺序
        sampled_conv_layers = [all_conv_layers[idx] for idx in indices]

    # 5. 仅为抽样选中的精简层节点注册前向钩子
    hook_manager = MultiLayerActivationHook()
    hook_handles = []
    for name, module in sampled_conv_layers:
        handle = module.register_forward_hook(hook_manager.get_hook_fn(name))
        hook_handles.append(handle)
            
    print(f"[*] 网络总计包含 {total_conv_count} 个卷积层。")
    print(f"[*] 选取 {len(hook_handles)} 层进行可视化展示。")
    print("-" * 60)
    
    try:
        # 6. 获取输入源并保存参考原图
        if use_real_image:
            val_split = config.get('val_split', 0.1) if config.get('val_split', 0.0) > 0.0 else 0.1
            input_tensor, view_img, label_idx = get_real_cifar_sample(val_split=val_split)
            
            cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            class_name = cifar10_classes[label_idx] if label_idx < len(cifar10_classes) else f"class_{label_idx}"
            
            plt.figure(figsize=(3, 3))
            plt.imshow(view_img)  
            plt.title(f"Real Input: {class_name}", weight='bold')
            plt.axis('off')
            ref_img_name = 'network_feature_input_real.png'
            hierarchy_img_name = 'network_feature_hierarchy_real.png'
        else:
            input_tensor, view_img = create_structured_synthetic_input(
                in_channels=all_conv_layers[0][1].in_channels, height=32, width=32
            )
            plt.figure(figsize=(3, 3))
            plt.imshow(view_img, cmap='gray', vmin=0, vmax=1)
            plt.title("Synthetic Input Signal", weight='bold')
            plt.axis('off')
            ref_img_name = 'network_feature_input_synthetic.png'
            hierarchy_img_name = 'network_feature_hierarchy_synthetic.png'
            
        plt.savefig(os.path.join(save_figures_dir, ref_img_name), dpi=200, bbox_inches='tight')
        plt.close()
        
        # 7. 执行前向传播
        with torch.no_grad():
            _ = model(input_tensor)
            
        # 8. 级联绘图排版
        captured_data = hook_manager.all_activated_features
        plot_network_feature_hierarchy(
            all_features=captured_data, 
            save_dir=save_figures_dir, 
            filename=hierarchy_img_name,
            channels_to_show=max_channels_per_layer
        )
        
        print("\n" + "=" * 60)
        print("     NETWORK FEATURE VISUALIZATION COMPLETED SUCCESSFULLY     ")
        print(f" Compact assets are saved safely under: {save_figures_dir}")
        print("=" * 60 + "\n")
        
    finally:
        # 9. 显式释放钩子句柄
        for handle in hook_handles:
            handle.remove()


if __name__ == "__main__":

    # target_pt_path: 指定 .pt 路径。若为 None 则自动检索 runs/ 下最新权重。
    # layers_to_show: 纵向展示的最大层数。推荐设为 5 或 6，图片高度非常美观。
    # channels_per_layer: 横向展示的特征通道数。
    # use_real_image: 是否使用真实彩色图像。
    target_pt_path = "runs/sgd_ce_gelu_ch64_b333_20260605_121836_epoch_100/best_model.pt"
    layers_to_show = 6       # 纵向层数截断控制器
    channels_per_layer = 4   # 横向通道截断控制器
    use_real_image = False 
    
    execute_network_visualization(
        model_path=target_pt_path, 
        max_layers_to_show=layers_to_show,
        max_channels_per_layer=channels_per_layer,
        use_real_image=use_real_image
    )
