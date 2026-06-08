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
num_workers = 4
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


def train(model, optimizer, criterion, train_loader, val_loader, model_name, lr, current_run_dir, epochs_n=20):
    model.to(device)
    
    # 通过 dir=current_run_dir 将 wandb 文件夹重定向至当前实验根目录下
    wandb.init(
        project="VGG_Optimization_Landscape",
        name=f"{model_name}_Simplified_lr_{lr}_{datetime.datetime.now().strftime('%m%d_%H%M')}",
        dir=current_run_dir, 
        config={
            "model_name": model_name,
            "learning_rate": lr,
            "epochs": epochs_n,
            "batch_size": batch_size,
            "seed": seed_val,
            "method": "Simplified"
        }
    )
    
    # 动态创建保存目录
    combo_dir = os.path.join(current_run_dir, f"{model_name}_SimplifiedMethod_lr_{lr}")
    history_dir = os.path.join(combo_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)

    losses_list = []
    grads = []
    best_val_accuracy = 0.0 
    batches_n = len(train_loader)

    for epoch in range(epochs_n):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{model_name} | LR: {lr}] Epoch {epoch+1:02d}/{epochs_n}", unit='batch')
        epoch_loss_sum = 0.0
        correct_train_preds = 0
        total_train_samples = 0
        
        loss_list = []  
        grad = []       

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 前向传播与 Loss 计算
            prediction = model(x)
            loss = criterion(prediction, y)
            current_loss_val = loss.item()
            
            loss_list.append(current_loss_val)
            epoch_loss_sum += current_loss_val
            
            # 统计训练集准确率相关数据 (在线计算，避免重复扫一遍 Dataset)
            _, train_predicted = torch.max(prediction.data, 1)
            total_train_samples += y.size(0)
            correct_train_preds += (train_predicted == y).sum().item()
            
            # 反向传播与特定层梯度观测
            loss.backward()
            
            if hasattr(model, 'classifier') and len(model.classifier) > 4 and model.classifier[4].weight.grad is not None:
                grad_norm = torch.norm(model.classifier[4].weight.grad).item()
                grad.append(grad_norm)
            else:
                grad.append(0.0)

            # 同步上传 Step 级别的指标至 wandb
            wandb.log({
                "step_loss": current_loss_val
            })

            optimizer.step()
            pbar.set_postfix(batch_loss=f"{current_loss_val:.4f}")

        losses_list.append(loss_list)
        grads.append(grad)

        # 验证集评估
        model.eval()
        val_loss_sum = 0.0
        correct_val_preds = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_prediction = model(x_val)
                v_loss = criterion(val_prediction, y_val)
                val_loss_sum += v_loss.item()
                
                _, val_predicted = torch.max(val_prediction.data, 1)
                total_val_samples += y_val.size(0)
                correct_val_preds += (val_predicted == y_val).sum().item()
        
        # 计算 Epoch 级别的最终指标
        train_loss = epoch_loss_sum / batches_n
        train_acc = correct_train_preds / total_train_samples
        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct_val_preds / total_val_samples

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc*100:.2f}%")

        # 将 Epoch 级别的汇总指标同步更新至 wandb
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": train_loss,
            "epoch_train_acc": train_acc,
            "epoch_val_loss": val_loss,
            "epoch_val_acc": val_acc
        })

        # 保存最佳 Checkpoint
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"--> Saved best checkpoint with Val Accuracy: {best_val_accuracy*100:.2f}%")
            
            metadata = {
                'model_name': model_name,
                'learning_rate': lr,
                'epoch': epoch + 1,
                'seed': seed_val,
                'best_val_accuracy': best_val_accuracy,
            }
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metadata': metadata
            }
            checkpoint_path = os.path.join(combo_dir, 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            
        print("-" * 50)

    # 展平数据并保存探索曲线原始文本数据
    flat_loss = [step_loss for epoch_loss in losses_list for step_loss in epoch_loss]
    flat_grad = [step_grad for epoch_grad in grads for step_grad in epoch_grad]
    
    np.savetxt(os.path.join(history_dir, 'step_losses.txt'), flat_loss, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'step_grads.txt'), flat_grad, fmt='%.6f')
    
    # 额外导出包含具体超参的独立元数据 JSON 文件
    total_params = sum(p.numel() for p in model.parameters())
    full_metadata = {
        "experiment_timestamp": timestamp,
        "model_architecture": model_name,
        "total_parameters": total_params,
        "hyperparameters": {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs_n,
            "random_seed": seed_val,
            "optimizer": "SGD",
            "criterion": "CrossEntropyLoss"
        },
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        },
        "final_metrics": {
            "best_val_accuracy": round(best_val_accuracy, 6),
            "final_epoch_train_accuracy": round(train_acc, 6),
            "final_epoch_val_accuracy": round(val_acc, 6)
        }
    }
    with open(os.path.join(combo_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=4, ensure_ascii=False)

    wandb.finish()
    return flat_loss


def plot_loss_landscape(min_c, max_c, min_c_bn, max_c_bn, figures_path, window_size=20):
    def moving_average(interval, window=window_size):
        w = np.ones(int(window)) / float(window)
        return np.convolve(interval, w, 'same')

    plt.figure(figsize=(11, 7), dpi=150)
    ax = plt.gca()
    ax.set_facecolor('#F4F6F9')
    plt.grid(True, color='white', linestyle='-', linewidth=1.5)

    color_vgg_line, color_vgg_fill = '#386B52', '#8EBC94'
    color_bn_line, color_bn_fill = '#9E2A2B', '#E29595'

    min_vgg_s = moving_average(min_c)
    max_vgg_s = moving_average(max_c)
    min_bn_s = moving_average(min_c_bn)
    max_bn_s = moving_average(max_c_bn)

    pad = window_size // 2
    steps = np.arange(len(min_vgg_s))[pad:-pad]

    # 图层一：Standard VGG
    plt.fill_between(steps, min_vgg_s[pad:-pad], max_vgg_s[pad:-pad], 
                     color=color_vgg_fill, alpha=0.45, label='Standard VGG')
    plt.plot(steps, min_vgg_s[pad:-pad], color=color_vgg_line, linewidth=1.0, alpha=0.7)
    plt.plot(steps, max_vgg_s[pad:-pad], color=color_vgg_line, linewidth=1.0, alpha=0.7)

    # 图层二：VGG + BatchNorm
    plt.fill_between(steps, min_bn_s[pad:-pad], max_bn_s[pad:-pad], 
                     color=color_bn_fill, alpha=0.45, label='VGG + BatchNorm')
    plt.plot(steps, min_bn_s[pad:-pad], color=color_bn_line, linewidth=1.0, alpha=0.7)
    plt.plot(steps, max_bn_s[pad:-pad], color=color_bn_line, linewidth=1.0, alpha=0.7)

    plt.title('Loss Landscape Smoothness (Simplified Method)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Optimization Steps', fontsize=12, labelpad=8)
    plt.ylabel('Loss Range across LR Pool', fontsize=12, labelpad=8)
    
    # ax.set_yscale('log')
        
    plt.legend(fontsize=11, loc='upper right', frameon=False)
    plt.xlim(steps[0], steps[-1])
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    final_fig_path = os.path.join(figures_path, 'loss_landscape_comparison.png')
    plt.tight_layout()
    plt.savefig(final_fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"[成功] 成果对比图已保存在: {final_fig_path}")


# 主程序入口
if __name__ == '__main__':
    print(f"正在使用的物理设备 (Device): {device}")
    if torch.cuda.is_available():
        print(f"显卡型号 (GPU Name): {torch.cuda.get_device_name(0)}")
    
    # 建立独立实验总目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(os.getcwd(), 'runs', 'task2', f"VGG_Simplified_Landscape_{timestamp}")
    figures_path = os.path.join(current_run_dir, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    print(f"实验根目录已建立: {current_run_dir}\n")

    val_split = 0.1
    train_loader = get_cifar_loader(batch_size=batch_size, train=True, val_split=val_split, is_val=False, shuffle=True, num_workers=num_workers)
    val_loader = get_cifar_loader(batch_size=batch_size, train=True, val_split=val_split, is_val=True, shuffle=False, num_workers=num_workers)
    print(f"数据加载器初始化成功：训练批次数量 {len(train_loader)}，验证批次数量 {len(val_loader)}")

    # DataLoader 状态自检流程
    for X, y in train_loader:
        print("="*50)
        print("【DataLoader 状态检查】")
        print(f"特征矩阵 X 维度: {X.shape} | 标签向量 y 维度: {y.shape}")
        print("="*50)
        
        img = X[0].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.title(f"Sample Label ID: {y[0].item()}")
        plt.axis('off')
        
        sample_save_path = os.path.join(figures_path, 'dataloader_sample.png')
        plt.savefig(sample_save_path)
        plt.close()
        print(f"样本图片保存至: {sample_save_path}\n")
        break

    epo = 20
    lr_list = [1e-1, 2e-1, 1e-2, 5e-2]

    vgg_loss_pool = []
    vgg_bn_loss_pool = []

    # 阶段一：标准 VGG_A 测算
    print("="*60)
    print(" 阶段一：开始训练基础 VGG_A 模型（Without BN）...")
    print("="*60)
    for lr in lr_list:
        set_random_seeds(seed_value=seed_val, device=device)
        model = VGG_A()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        criterion = nn.CrossEntropyLoss()
        
        flat_loss = train(
            model=model, optimizer=optimizer, criterion=criterion, 
            train_loader=train_loader, val_loader=val_loader, 
            model_name="VGG_A", lr=lr, current_run_dir=current_run_dir, epochs_n=epo
        )
        vgg_loss_pool.append(flat_loss)

    # 阶段二：带 BatchNorm 的 VGG_A 测算
    print("\n" + "="*60)
    print(" 阶段二：开始训练带 BN 的 VGG_A_BatchNorm 模型...")
    print("="*60)
    for lr in lr_list:
        set_random_seeds(seed_value=seed_val, device=device)
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        criterion = nn.CrossEntropyLoss()
        
        flat_loss = train(
            model=model, optimizer=optimizer, criterion=criterion, 
            train_loader=train_loader, val_loader=val_loader, 
            model_name="VGG_A_BatchNorm", lr=lr, current_run_dir=current_run_dir, epochs_n=epo
        )
        vgg_bn_loss_pool.append(flat_loss)

    # 阶段三：绘图
    print("\n" + "="*60)
    print(" 阶段三：生成对比图...")
    print("="*60)
    vgg_loss_pool = np.array(vgg_loss_pool)       
    vgg_bn_loss_pool = np.array(vgg_bn_loss_pool) 

    min_curve = np.min(vgg_loss_pool, axis=0)
    max_curve = np.max(vgg_loss_pool, axis=0)
    min_curve_bn = np.min(vgg_bn_loss_pool, axis=0)
    max_curve_bn = np.max(vgg_bn_loss_pool, axis=0)

    np.savetxt(os.path.join(current_run_dir, 'landscape_vgg_min_curve.txt'), min_curve, fmt='%.6f')
    np.savetxt(os.path.join(current_run_dir, 'landscape_vgg_max_curve.txt'), max_curve, fmt='%.6f')
    np.savetxt(os.path.join(current_run_dir, 'landscape_vgg_bn_min_curve.txt'), min_curve_bn, fmt='%.6f')
    np.savetxt(os.path.join(current_run_dir, 'landscape_vgg_bn_max_curve.txt'), max_curve_bn, fmt='%.6f')

    plot_loss_landscape(min_curve, max_curve, min_curve_bn, max_curve_bn, figures_path=figures_path)
