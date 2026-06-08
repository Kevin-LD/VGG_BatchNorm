import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

sys.path.append(os.getcwd())

from VGG_Loss_Landscape_Original import plot_landscape_metric


def main(exp_dir):
    print("=" * 60)
    print("       VGG LOSS LANDSCAPE HISTORICAL DATA PLOTTER       ")
    print("=" * 60)
    print(f"Target Experiment Root: {exp_dir}")

    vgg_history_dir = None
    bn_history_dir = None

    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"[错误] 指定的实验总目录不存在: '{exp_dir}'")

    for item in os.listdir(exp_dir):
        full_path = os.path.join(exp_dir, item)
        if os.path.isdir(full_path):
            if item.startswith("VGG_A_OriginalMethod_lr_"):
                vgg_history_dir = os.path.join(full_path, "history")
            elif item.startswith("VGG_A_BatchNorm_OriginalMethod_lr_"):
                bn_history_dir = os.path.join(full_path, "history")

    # 安全检查
    if not vgg_history_dir or not os.path.exists(vgg_history_dir):
        raise FileNotFoundError(f"[错误] 未能在目录下定位到 Standard VGG 的历史文本数据文件夹。")
    if not bn_history_dir or not os.path.exists(bn_history_dir):
        raise FileNotFoundError(f"[错误] 未能在目录下定位到 VGG+BatchNorm 的历史文本数据文件夹。")

    # 确定图片输出目录
    figures_path = os.path.join(exp_dir, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    print(f"[解析] 检测到 Standard 数据源: {vgg_history_dir}")
    print(f"[解析] 检测到 BatchNorm 数据源: {bn_history_dir}")
    print("--> 正在从本地磁盘反序列化 TXT 历史矩阵...")

    # 从原始文本中一键加载一维数组
    try:
        loss_vgg_min = np.loadtxt(os.path.join(vgg_history_dir, 'original_landscape_min.txt'))
        loss_vgg_max = np.loadtxt(os.path.join(vgg_history_dir, 'original_landscape_max.txt'))
        loss_bn_min = np.loadtxt(os.path.join(bn_history_dir, 'original_landscape_min.txt'))
        loss_bn_max = np.loadtxt(os.path.join(bn_history_dir, 'original_landscape_max.txt'))
        
        pred_vgg_min = np.loadtxt(os.path.join(vgg_history_dir, 'grad_predict_min.txt'))
        pred_vgg_max = np.loadtxt(os.path.join(vgg_history_dir, 'grad_predict_max.txt'))
        pred_bn_min = np.loadtxt(os.path.join(bn_history_dir, 'grad_predict_min.txt'))
        max_bn_max = np.loadtxt(os.path.join(bn_history_dir, 'grad_predict_max.txt')) # 映射你的副本数据
        
        beta_vgg_max = np.loadtxt(os.path.join(vgg_history_dir, 'beta_smooth_max.txt'))
        beta_bn_max = np.loadtxt(os.path.join(bn_history_dir, 'beta_smooth_max.txt'))
    except Exception as e:
        raise IOError(f"[错误] 加载原始文本数据失败，请确认历史数据是否完整保存。 详细报错: {e}")

    print("[加载完成] 文本矩阵提取完毕。开始并行渲染高精图表...")
    
    # 调整 window_size
    LOSS_LANDSCAPE_W_SIZE = 20
    GRADIENT_PREDICTIVENESS_W_SIZE = 1
    EFFECITVE_BETA_SMOOTHNESS_W_SIZE = 1
    
    # 阶段三：调用绘图引擎重新生成最终的三大景观对比图
    print("\n" + "-" * 50)
    print(" 绘图流 1/3: 重新生成 Loss Landscape Smoothness 条带图...")
    plot_landscape_metric(
        vgg_data=(loss_vgg_min, loss_vgg_max), 
        bn_data=(loss_bn_min, loss_bn_max),
        metric_title='Loss Landscape Smoothness',
        y_label='Loss Range along Gradient Direction',
        file_name='loss_landscape.png',
        figures_path=figures_path,
        window_size=LOSS_LANDSCAPE_W_SIZE,
        as_line=False
    )

    print("\n 绘图流 2/3: 重新生成 Gradient Predictiveness 条带图...")
    plot_landscape_metric(
        vgg_data=(pred_vgg_min, pred_vgg_max), 
        bn_data=(pred_bn_min, max_bn_max),
        metric_title='Gradient Predictiveness ($L_2$ Distance of Gradients)',
        y_label=r'$\|\nabla L(W) - \nabla L(W_{perturbed})\|_2^2$',
        file_name='gradient_predictiveness.png',
        figures_path=figures_path,
        window_size=GRADIENT_PREDICTIVENESS_W_SIZE,
        as_line=False
    )

    print("\n 绘图流 3/3: 重新生成 Effective Beta-Smoothness 曲线图...")
    plot_landscape_metric(
        vgg_data=beta_vgg_max, 
        bn_data=beta_bn_max,
        metric_title=r'Effective $\beta$-Smoothness along Optimization Path',
        y_label=r'$\max_{\alpha} (\|\Delta \nabla L\|_2 / \alpha)$',
        file_name='beta_smoothness.png',
        figures_path=figures_path,
        window_size=EFFECITVE_BETA_SMOOTHNESS_W_SIZE,
        as_line=True
    )

    print("\n" + "=" * 60)
    print(f"图片保存至:\n{os.path.abspath(figures_path)}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    exp_dir = "runs/task2/VGG_OriginalPaper_Landscape_20260608_162958"
    
    main(exp_dir)
