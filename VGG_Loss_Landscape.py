import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import datetime
import json

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# Constants (parameters) initialization
device_id = [0]
num_workers = 4
batch_size = 128
seed_val = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_accuracy(model, data_loader):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():  
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            _, predicted_classes = torch.max(prediction, 1)
            total += y.size(0)
            correct += (predicted_classes == y).sum().item()
    return correct / total


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, optimizer, criterion, train_loader, val_loader, model_name, lr, scheduler=None, epochs_n=100):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    
    # 为当前组合动态开辟独立的实验子目录与子 history 目录
    combo_dir = os.path.join(current_run_dir, f"{model_name}_lr_{lr}")
    history_dir = os.path.join(combo_dir, 'history')
    os.makedirs(combo_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    best_model_path = os.path.join(combo_dir, 'best_model.pth')

    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  
        grad = []       
        learning_curve[epoch] = 0  

        pbar = tqdm(train_loader, desc=f"[{model_name} | LR: {lr}] Epoch {epoch+1:02d}/{epochs_n}", unit='batch')
        for data in pbar:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            current_loss_val = loss.item()
            loss_list.append(current_loss_val)
            learning_curve[epoch] += current_loss_val
            
            loss.backward()
            
            if model.classifier[4].weight.grad is not None:
                grad_norm = torch.norm(model.classifier[4].weight.grad).item()
                grad.append(grad_norm)
            else:
                grad.append(0.0)

            optimizer.step()
            pbar.set_postfix(batch_loss=f"{current_loss_val:.4f}")

        losses_list.append(loss_list)
        grads.append(grad)
        
        # 测试精度与最佳模型留存
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Val Accuracy:   {val_acc*100:.2f}%")
        
        if val_acc > max_val_accuracy:
            print(f"--> Saved best checkpoint with Val Accuracy: {val_acc*100:.2f}%")
            max_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)

    # 展平数据
    flat_loss = [step_loss for epoch_loss in losses_list for step_loss in epoch_loss]
    flat_grad = [step_grad for epoch_grad in grads for step_grad in epoch_grad]
    
    np.savetxt(os.path.join(history_dir, 'step_losses.txt'), flat_loss, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'step_grads.txt'), flat_grad, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'epoch_losses.txt'), [l/batches_n for l in learning_curve], fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'epoch_train_acc.txt'), train_accuracy_curve, fmt='%.6f')
    np.savetxt(os.path.join(history_dir, 'epoch_val_acc.txt'), val_accuracy_curve, fmt='%.6f')

    # 计算模型的总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 构建完整的元数据字典，并在权重文件夹下保存为可读性极佳的 json 文件
    metadata = {
        "experiment_timestamp": timestamp,
        "model_architecture": model_name,
        "total_parameters": total_params,
        "hyperparameters": {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs_n,
            "random_seed": seed_val,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss"
        },
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        },
        "final_metrics": {
            "best_val_accuracy": round(max_val_accuracy, 6),
            "final_epoch_train_accuracy": round(train_accuracy_curve[-1], 6),
            "final_epoch_val_accuracy": round(val_accuracy_curve[-1], 6)
        }
    }
    
    metadata_json_path = os.path.join(combo_dir, 'metadata.json')
    with open(metadata_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    print(f"组合完成 -> 权重、历史记录与元数据保存至: {combo_dir}")

    return flat_loss

def plot_loss_landscape(min_c, max_c, min_c_bn, max_c_bn, window_size=50):
    def moving_average(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    # 对四条边界曲线进行平滑处理，消除 step-level 的毛刺
    min_c_smooth = moving_average(min_c, window_size)
    max_c_smooth = moving_average(max_c, window_size)
    min_c_bn_smooth = moving_average(min_c_bn, window_size)
    max_c_bn_smooth = moving_average(max_c_bn, window_size)
    

    pad = window_size // 2
    steps = np.arange(len(min_c_smooth))[pad:-pad]

    plt.figure(figsize=(12, 7.5), dpi=150)
    
    ax = plt.gca()
    ax.set_facecolor('#F0F2F6')
    plt.grid(True, color='white', linestyle='-', linewidth=1.5) # 白色网格线
    

    color_vgg_line = '#4A7C59'
    color_vgg_fill = '#95C7AE'
    
    color_bn_line = '#A63A50'
    color_bn_fill = '#D68C96'

    # 图层一：Standard VGG (Without BN)
    plt.fill_between(steps, min_c_smooth[pad:-pad], max_c_smooth[pad:-pad], 
                     color=color_vgg_fill, alpha=0.5, label='Standard VGG')
    plt.plot(steps, min_c_smooth[pad:-pad], color=color_vgg_line, linewidth=1.2, alpha=0.8)
    plt.plot(steps, max_c_smooth[pad:-pad], color=color_vgg_line, linewidth=1.2, alpha=0.8)

    # 图层二：Standard VGG + BatchNorm (With BN)
    plt.fill_between(steps, min_c_bn_smooth[pad:-pad], max_c_bn_smooth[pad:-pad], 
                     color=color_bn_fill, alpha=0.5, label='Standard VGG + BatchNorm')
    plt.plot(steps, min_c_bn_smooth[pad:-pad], color=color_bn_line, linewidth=1.2, alpha=0.8)
    plt.plot(steps, max_c_bn_smooth[pad:-pad], color=color_bn_line, linewidth=1.2, alpha=0.8)


    plt.title('Loss Landscape', fontsize=16, pad=15, fontweight='medium')
    plt.xlabel('Steps', fontsize=13, labelpad=10)
    plt.ylabel('Loss Landscape', fontsize=13, labelpad=10)
    
    # 优化图例：去掉边框、放大字体，使其融入背景
    plt.legend(fontsize=14, loc='upper right', frameon=False)
    
    # 美化坐标轴边界
    plt.xlim(steps[0] - 100, steps[-1] + 100)
    plt.tick_params(colors='#4A4A4A', labelsize=10) # 轴标签改用深灰，比纯黑更高级
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    final_fig_path = os.path.join(figures_path, 'loss_landscape_comparison.png')
    plt.tight_layout()
    plt.savefig(final_fig_path, dpi=300, facecolor='white') # 确保边缘导出时不留黑边
    plt.close()
    
    print(f"\n成果图保存在: {final_fig_path}")

if __name__ == '__main__':
    print(f"正在使用的物理设备 (Device): {device}")
    if torch.cuda.is_available():
        print(f"显卡型号 (GPU Name): {torch.cuda.get_device_name(0)}")
    
    # 创建本次实验的总根目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(os.getcwd(), 'runs', 'task2', f"VGG_Optimization_Exp_{timestamp}")
    figures_path = os.path.join(current_run_dir, 'figures')

    os.makedirs(current_run_dir, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    print(f"总实验根目录已建立: {current_run_dir}\n")


    train_loader = get_cifar_loader(train=True, augment=False)
    val_loader = get_cifar_loader(train=False)

    for X, y in train_loader:
        print("="*50)
        print("【DataLoader 状态检查】")
        print(f"特征矩阵 X 维度: {X.shape} | 标签向量 y 维度: {y.shape}")
        print("="*50)
        
        img = X[0].permute(1, 2, 0).numpy() # [C, H, W] -> [H, W, C]
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
    lr_list = [1e-3, 2e-3, 1e-4, 5e-4]

    vgg_loss_pool = []
    vgg_bn_loss_pool = []

    print("="*60)
    print("阶段一：开始训练基础 VGG_A 模型（Without BN）...")
    print("="*60)
    for lr in lr_list:
        set_random_seeds(seed_value=seed_val, device=device)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 调用训练，内部会自动处理子目录、元数据 JSON 和 history 文件夹
        flat_loss = train(model, optimizer, criterion, train_loader, val_loader, 
                        model_name="VGG_A", lr=lr, epochs_n=epo)
        vgg_loss_pool.append(flat_loss)

    print("\n" + "="*60)
    print("阶段二：开始训练带 BN 的 VGG_A_BatchNorm 模型...")
    print("="*60)
    for lr in lr_list:
        set_random_seeds(seed_value=seed_val, device=device)
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        flat_loss = train(model, optimizer, criterion, train_loader, val_loader, 
                        model_name="VGG_A_BatchNorm", lr=lr, epochs_n=epo)
        vgg_bn_loss_pool.append(flat_loss)


    # 全局汇总极值计算并保存到根目录
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

    plot_loss_landscape(min_curve, max_curve, min_curve_bn, max_curve_bn)
