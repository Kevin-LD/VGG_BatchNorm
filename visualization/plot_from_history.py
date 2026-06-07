import os
import sys
import numpy as np

sys.path.append(os.getcwd())

try:
    import VGG_Loss_Landscape
    from VGG_Loss_Landscape import plot_loss_landscape
except ImportError:
    print("错误：未能在当前工作目录下找到 VGG_Loss_Landscape.py 文件。")
    print("请确保在包含 VGG_Loss_Landscape.py 的正确路径下运行此脚本。")
    sys.exit(1)

def main(exp_dir):    
    if not os.path.exists(exp_dir):
        print(f"错误：指定的实验目录不存在 -> '{exp_dir}'")
        return

    # 自动构建 4 个 curve 的 txt 历史记录文件路径
    min_vgg_path = os.path.join(exp_dir, 'landscape_vgg_min_curve.txt')
    max_vgg_path = os.path.join(exp_dir, 'landscape_vgg_max_curve.txt')
    min_bn_path = os.path.join(exp_dir, 'landscape_vgg_bn_min_curve.txt')
    max_bn_path = os.path.join(exp_dir, 'landscape_vgg_bn_max_curve.txt')
    
    # 检查文件完整性
    required_files = [min_vgg_path, max_vgg_path, min_bn_path, max_bn_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误：找不到必要的历史曲线文件 -> {file_path}")
            print("请检查该实验目录下的数据是否完整。")
            return

    # 自动读取 4 个 curve 的 txt 历史记录
    print("正在从指定子目录加载历史文本数据...")
    min_curve = np.loadtxt(min_vgg_path)
    max_curve = np.loadtxt(max_vgg_path)
    min_curve_bn = np.loadtxt(min_bn_path)
    max_curve_bn = np.loadtxt(max_bn_path)
    
    print(f"成功加载数据！总时间步长 (Total Training Steps): {len(min_curve)}")

    target_figures_path = os.path.join(exp_dir, 'figures')
    os.makedirs(target_figures_path, exist_ok=True)
    VGG_Loss_Landscape.figures_path = target_figures_path

    try:
        window_size_config = 30
        plot_loss_landscape(min_curve, max_curve, min_curve_bn, max_curve_bn, window_size=window_size_config)
    except TypeError:
        plot_loss_landscape(min_curve, max_curve, min_curve_bn, max_curve_bn)
        
    print("\n" + "="*60)
    print(f"成果图已保存在目标实验目录下：")
    print(f"图片路径: {os.path.join(target_figures_path, 'loss_landscape_comparison.png')}")
    print("="*60)

if __name__ == '__main__':
    exp_dir = "runs/task2/VGG_Optimization_Exp_20260607_101350"
    main(exp_dir)
