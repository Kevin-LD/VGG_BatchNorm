import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
import datetime
import json
import wandb

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader


# 1. 基础配置与随机种子初始化
batch_size = 128
seed_val = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 核心训练与三项地形指标同步测算
def train_and_explore_landscape(model, optimizer, criterion, train_loader, model_name, lr, current_run_dir, epochs_n=20):
    model.to(device)
    
    # 【此处已更新】通过 dir=current_run_dir 将 wandb 文件夹重定向至当前实验根目录下
    wandb.init(
        project="VGG_Optimization_Landscape",
        name=f"{model_name}_lr_{lr}_{datetime.datetime.now().strftime('%m%d_%H%M')}",
        dir=current_run_dir, 
        config={
            "model_name": model_name,
            "learning_rate": lr,
            "epochs": epochs_n,
            "batch_size": batch_size,
            "seed": seed_val
        }
    )
    
    # 用于记录全优化流中，每一个 step 沿梯度方向探测到的三项指标边界
    landscape_max_curve, landscape_min_curve = [], []
    predictiveness_max_curve, predictiveness_min_curve = [], []
    beta_max_curve = []  # 完美还原论文：Beta Smoothness 只需要记录每步的最大值
    
    # 动态创建保存目录
    combo_dir = os.path.join(current_run_dir, f"{model_name}_OriginalMethod_lr_{lr}")
    history_dir = os.path.join(combo_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)

    best_loss = float('inf')  # 用于追踪最佳模型

    for epoch in range(epochs_n):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1:02d}/{epochs_n}", unit='batch')
        epoch_loss_sum = 0.0
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 计算当前权重下的原生前向传播与 Loss
            prediction = model(x)
            loss = criterion(prediction, y)
            current_loss_val = loss.item()
            epoch_loss_sum += current_loss_val
            
            # 反向传播算出当前方向的原始梯度 g
            loss.backward()
            
            # 原论文做法：沿着当前梯度方向进行多点扰动采样与三项指标测算
            original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            g_flat = torch.cat([g.detach().view(-1) for g in original_grads])
            grad_norm = g_flat.norm().item()
            
            if grad_norm > 1e-8:
                # 定义探测步长 alpha 的范围，在 [-0.02, 0.02] 之间均匀采样 9 个点
                alphas = np.linspace(-0.02, 0.02, 9)
                perturbed_losses = []
                perturbed_grad_predictiveness = []
                perturbed_betas = []
                
                # 开始进行影子扰动测试
                for alpha in alphas:
                    # 临时改写参数: W_perturbed = W + alpha * (g / ||g||)
                    with torch.no_grad():
                        idx = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                p.add_(original_grads[idx], alpha=(alpha / grad_norm))
                                idx += 1
                    
                    # 开启梯度清零，用于计算扰动处的独立一阶导数
                    model.zero_grad()
                    perturbed_output = model(x)
                    p_loss = criterion(perturbed_output, y)
                    perturbed_losses.append(p_loss.item())
                    
                    # 在新位置反向传播，采集影子梯度
                    p_loss.backward()
                    g_perturbed_flat = torch.cat([p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None])
                    
                    # 指标二：计算新旧梯度的 L2 距离平方 (Gradient Predictiveness)
                    grad_diff_norm = torch.norm(g_flat - g_perturbed_flat).item()
                    perturbed_grad_predictiveness.append(grad_diff_norm ** 2)
                    
                    # 指标三：计算有效平滑度比值 (Effective Beta Smoothness)
                    if abs(alpha) > 1e-8:
                        beta_smoothness = grad_diff_norm / abs(alpha)
                        perturbed_betas.append(beta_smoothness)
                    
                    # 精准还原参数权重: W = W_perturbed - alpha * (g / ||g||)
                    with torch.no_grad():
                        idx = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                p.add_(original_grads[idx], alpha=(-alpha / grad_norm))
                                idx += 1
                
                # 提取当前 Step 扰动域内的极值边界
                step_loss_max, step_loss_min = max(perturbed_losses), min(perturbed_losses)
                step_pred_max, step_pred_min = max(perturbed_grad_predictiveness), min(perturbed_grad_predictiveness)
                step_beta_max = max(perturbed_betas)  # 还原论文：取最大值
            else:
                # 若无梯度，各指标均退化为基准点数据
                step_loss_max = step_loss_min = current_loss_val
                step_pred_max = step_pred_min = 0.0
                step_beta_max = 0.0
            
            # 回填由于采样被清空的原始梯度，确保主优化步方向绝对正确
            with torch.no_grad():
                idx = 0
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.copy_(original_grads[idx])
                        idx += 1
            
            # 记录全优化流数据
            landscape_max_curve.append(step_loss_max)
            landscape_min_curve.append(step_loss_min)
            predictiveness_max_curve.append(step_pred_max)
            predictiveness_min_curve.append(step_pred_min)
            beta_max_curve.append(step_beta_max)

            # 同步上传至 wandb
            wandb.log({
                "loss": current_loss_val,
                "loss_gap": step_loss_max - step_loss_min,
                "grad_predictiveness_max": step_pred_max,
                "grad_predictiveness_min": step_pred_min,
                "beta_smoothness_max": step_beta_max
            })

            # 执行真正的参数更新步
            optimizer.step()
            pbar.set_postfix(loss=f"{current_loss_val:.4f}", gap=f"{(step_loss_max-step_loss_min):.4f}")

        # 检查并保存最佳模型权重
        epoch_loss_avg = epoch_loss_sum / len(train_loader)
        if epoch_loss_avg < best_loss:
            best_loss = epoch_loss_avg
            metadata = {
                'model_name': model_name,
                'learning_rate': lr,
                'epoch': epoch + 1,
                'seed': seed_val,
                'best_train_loss': best_loss,
                'save_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metadata': metadata
            }
            checkpoint_path = os.path.join(combo_dir, 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)

    # 保存探索曲线原始文本数据
    np.savetxt(os.path.join(history_dir, 'original_landscape_max.txt'), landscape_max_curve, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'original_landscape_min.txt'), landscape_min_curve, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'grad_predict_max.txt'), predictiveness_max_curve, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'grad_predict_min.txt'), predictiveness_min_curve, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'beta_smooth_max.txt'), beta_max_curve, fmt='%.6f')
    
    wandb.finish()  # 结束当前模型的 wandb 进程
    
    return (np.array(landscape_min_curve), np.array(landscape_max_curve)), \
           (np.array(predictiveness_min_curve), np.array(predictiveness_max_curve)), \
           np.array(beta_max_curve)


def plot_landscape_metric(vgg_data, bn_data, metric_title, y_label, file_name, figures_path, window_size=21, as_line=False):
    def moving_average(interval, window=window_size):
        w = np.ones(int(window)) / float(window)
        return np.convolve(interval, w, 'same')

    plt.figure(figsize=(11, 7), dpi=150)
    ax = plt.gca()
    ax.set_facecolor('#F4F6F9')
    plt.grid(True, color='white', linestyle='-', linewidth=1.5)

    color_vgg_line, color_vgg_fill = '#386B52', '#8EBC94'
    color_bn_line, color_bn_fill = '#9E2A2B', '#E29595'

    if as_line:
        # 还原论文：单线条图
        vgg_s = moving_average(vgg_data)
        bn_s = moving_average(bn_data)
        pad = window_size // 2
        steps = np.arange(len(vgg_s))[pad:-pad]

        plt.plot(steps, vgg_s[pad:-pad], color=color_vgg_line, linewidth=2.2, label='Standard VGG (Original Method)')
        plt.plot(steps, bn_s[pad:-pad], color=color_bn_line, linewidth=2.2, label='VGG + BatchNorm (Original Method)')
    else:
        # 解包带状范围数据
        min_vgg, max_vgg = vgg_data
        min_bn, max_bn = bn_data

        min_vgg_s = moving_average(min_vgg)
        max_vgg_s = moving_average(max_vgg)
        min_bn_s = moving_average(min_bn)
        max_bn_s = moving_average(max_bn)

        pad = window_size // 2
        steps = np.arange(len(min_vgg_s))[pad:-pad]

        # Standard VGG 带状图
        plt.fill_between(steps, min_vgg_s[pad:-pad], max_vgg_s[pad:-pad], 
                         color=color_vgg_fill, alpha=0.45, label='Standard VGG (Original Method)')
        plt.plot(steps, min_vgg_s[pad:-pad], color=color_vgg_line, linewidth=1.0, alpha=0.7)
        plt.plot(steps, max_vgg_s[pad:-pad], color=color_vgg_line, linewidth=1.0, alpha=0.7)

        # VGG + BatchNorm 带状图
        plt.fill_between(steps, min_bn_s[pad:-pad], max_bn_s[pad:-pad], 
                         color=color_bn_fill, alpha=0.45, label='VGG + BatchNorm (Original Method)')
        plt.plot(steps, min_bn_s[pad:-pad], color=color_bn_line, linewidth=1.0, alpha=0.7)
        plt.plot(steps, max_bn_s[pad:-pad], color=color_bn_line, linewidth=1.0, alpha=0.7)

    plt.title(metric_title, fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Optimization Steps', fontsize=12, labelpad=8)
    plt.ylabel(y_label, fontsize=12, labelpad=8)
    
    plt.legend(fontsize=11, loc='upper right', frameon=False)
    plt.xlim(steps[0], steps[-1])
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    final_fig_path = os.path.join(figures_path, file_name)
    plt.tight_layout()
    plt.savefig(final_fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"[成功] 成果图已保存在: {final_fig_path}")


# 主程序入口
if __name__ == '__main__':
    print(f"正在使用的物理设备 (Device): {device}")
    
    # 建立独立实验总目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(os.getcwd(), 'runs', 'task2', f"VGG_OriginalPaper_Landscape_{timestamp}")
    figures_path = os.path.join(current_run_dir, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    print(f"原论文实验根目录已建立: {current_run_dir}\n")

    # 获取数据加载器
    train_loader = get_cifar_loader(train=True, augment=False)

    # 设定固定的标准学习率与 Epoch 数
    epo = 20
    target_lr = 1e-3

    # 阶段一：标准 VGG_A 测算
    print("="*60)
    print(" 阶段一：测算 Standard VGG 的三种地形指标 (原论文方法)...")
    print("="*60)
    set_random_seeds(seed_value=seed_val, device=device)
    model_vgg = VGG_A()
    optimizer_vgg = torch.optim.Adam(model_vgg.parameters(), lr=target_lr)
    criterion = nn.CrossEntropyLoss()
    
    loss_vgg, pred_vgg, beta_vgg = train_and_explore_landscape(
        model=model_vgg, optimizer=optimizer_vgg, criterion=criterion, 
        train_loader=train_loader, model_name="VGG_A", lr=target_lr, 
        current_run_dir=current_run_dir, epochs_n=epo
    )

    # 阶段二：带 BatchNorm 的 VGG_A 测算
    print("\n" + "="*60)
    print(" 阶段二：测算 VGG_A_BatchNorm 的三种地形指标 (原论文方法)...")
    print("="*60)
    set_random_seeds(seed_value=seed_val, device=device)
    model_bn = VGG_A_BatchNorm()
    optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=target_lr)
    
    loss_bn, pred_bn, beta_bn = train_and_explore_landscape(
        model=model_bn, optimizer=optimizer_bn, criterion=criterion, 
        train_loader=train_loader, model_name="VGG_A_BatchNorm", lr=target_lr, 
        current_run_dir=current_run_dir, epochs_n=epo
    )

    # 阶段三：绘制对比图
    print("\n" + "="*60)
    print(" 阶段三：生成最终对比图...")
    print("="*60)
    
    # 1. 绘制 Loss Landscape
    plot_landscape_metric(
        vgg_data=loss_vgg, bn_data=loss_bn,
        metric_title='Loss Landscape Smoothness (NeurIPS 2018 Method)',
        y_label='Loss Range along Gradient Direction',
        file_name='original_paper_loss_landscape.png',
        figures_path=figures_path,
        as_line=False
    )
    
    # 2. 绘制 Gradient Predictiveness
    plot_landscape_metric(
        vgg_data=pred_vgg, bn_data=pred_bn,
        metric_title='Gradient Predictiveness ($L_2$ Distance of Gradients)',
        y_label=r'$\|\nabla L(W) - \nabla L(W_{perturbed})\|_2^2$',
        file_name='original_paper_gradient_predictiveness.png',
        figures_path=figures_path,
        as_line=False
    )
    
    # 3. 绘制 Effective Beta Smoothness
    plot_landscape_metric(
        vgg_data=beta_vgg, bn_data=beta_bn,
        metric_title='Effective $\beta$-Smoothness along Optimization Path',
        y_label=r'$\max_{\alpha} (\|\Delta \nabla L\|_2 / \alpha)$',
        file_name='original_paper_beta_smoothness.png',
        figures_path=figures_path,
        as_line=True
    )
