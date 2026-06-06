import os
import time
import json
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 尝试导入 wandb，若未安装则输出提示
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb package not found. Training will proceed without W&B logging.")

# 导入自定义的数据加载器与网络结构
from data.loaders import get_cifar_loader
from models.custom_cnn import CIFAR10ResidualNet, get_number_of_parameters

from utils.eval import eval
from utils.training import get_criterion, get_optimizer

def train_model(config, save_dir='runs', project_name='cifar10_task1', run_name=None):
    """
    独立训练核心函数。
    支持显式指定存储根目录、项目名称与运行批次名称，并在训练结束后返回最优指标字典。
    """
    # 1. 硬件设备准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 动态生成或使用指定的存储路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    blocks_str = "".join(map(str, config.get('num_blocks', [2, 2, 2])))
    
    # 若外部未指定运行名称，则基于当前配置和时间戳自动构建
    if run_name is None:
        run_name = f"{config['optimizer_type']}_{config['loss_type']}_{config['activation_type']}_ch{config['base_channels']}_b{blocks_str}_{timestamp}"
    
    # 将路径拼接至传入的 save_dir 中，实现目录的层级嵌套
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Experiment directory created at: {run_dir}")

    # 3. 初始化 Weights & Biases 监控
    if WANDB_AVAILABLE:
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            dir=run_dir
        )

    # 4. 数据加载器初始化
    val_split = config.get('val_split', 0.1)
    train_loader = get_cifar_loader(batch_size=config['batch_size'], train=True, val_split=val_split, is_val=False, shuffle=True, num_workers=4)
    val_loader = get_cifar_loader(batch_size=config['batch_size'], train=True, val_split=val_split, is_val=True, shuffle=False, num_workers=4)

    # 5. 模型实例化
    model = CIFAR10ResidualNet(
        base_channels=config['base_channels'],
        num_blocks=config.get('num_blocks', [2, 2, 2]),
        dropout_rate=config['dropout_rate'],
        activation_type=config['activation_type']
    ).to(device)
    
    total_params = get_number_of_parameters(model)
    print(f"Total Parameters: {total_params:,}")

    # 6. 损失函数与优化器配置
    criterion = get_criterion(config['loss_type'], config.get('label_smoothing', 0.0))
    optimizer = get_optimizer(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    history = []
    best_val_accuracy = 0.0 

    # 返回给外部调度器的核心信息
    best_run_summary = {
        'run_name': run_name,
        'run_dir': run_dir,
        'best_epoch': 0,
        'best_val_accuracy': 0.0,
        'corresponding_val_loss': float('inf'),
        'corresponding_train_loss': float('inf'),
        'corresponding_train_accuracy': 0.0,
        'total_parameters': total_params,
        'timestamp': timestamp
    }

    # 7. 核心训练循环
    print("Starting training...")
    try:
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()

            for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}"):
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            epoch_time = time.time() - start_time
            throughput = total / epoch_time

            train_loss = running_loss / total
            train_acc = correct / total

            # 在验证集上进行评估
            val_loss, val_acc = eval(model, val_loader, criterion, device)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            print(f"Epoch [{epoch}/{config['epochs']}] | Speed: {throughput:.1f} img/s")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc*100:.2f}%")

            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'throughput': throughput,
                'lr': current_lr
            }
            history.append(metrics)

            if WANDB_AVAILABLE:
                wandb.log(metrics)

            # 基于验证集最高准确率保存权重、元数据并更新外部返回字典
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                
                # 动态同步更新需要向外部返回的报告摘要
                best_run_summary.update({
                    'best_epoch': epoch,
                    'best_val_accuracy': val_acc,
                    'corresponding_val_loss': val_loss,
                    'corresponding_train_loss': train_loss,
                    'corresponding_train_accuracy': train_acc
                })

                metadata = {
                    'epoch': epoch,
                    'best_val_accuracy': best_val_accuracy,
                    'total_parameters': total_params,
                    'timestamp': timestamp,
                    'config': config,
                    'metrics_at_checkpoint': metrics
                }
                
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'metadata': metadata
                }
                
                checkpoint_path = os.path.join(run_dir, 'best_model.pt')
                torch.save(checkpoint, checkpoint_path)
                
                with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                    
                print(f"--> Saved best checkpoint with Val Accuracy: {best_val_accuracy*100:.2f}%")
            print("-" * 60)

    finally:
        # 8. 善后处理机制
        print("Training interrupted or completed. Saving logs...")
        if history:
            log_df = pd.DataFrame(history)
            log_df.to_csv(os.path.join(run_dir, 'training_log.csv'), index=False)
        
        if WANDB_AVAILABLE:
            wandb.finish()
            
        print(f"All saved logs and weights are safe in {run_dir}")
        
    # 将包含核心评估指标的字典返回给调用它的搜索脚本
    return best_run_summary


if __name__ == "__main__":
    # 实验超参数配置字典
    experiment_config = {
        'epochs': 100,
        'batch_size': 128,
        'lr': 0.01,
        
        # 验证集划分比例配置
        'val_split': 0.1,

        # 策略 4(a): 滤波器基准通道数 (可尝试 16, 32, 64)
        'base_channels': 64,
        
        # 策略 4(a): 网络深度配置 (各 Stage 堆叠的残差块数量，可尝试 [1, 1, 1], [2, 2, 2], [3, 3, 3])
        'num_blocks': [3, 3, 3],
        
        # dropout rate
        'dropout_rate': 0.3,

        # 策略 4(c): 激活函数类型 (可选择 'relu', 'leaky_relu', 'gelu')
        'activation_type': 'relu',
        
        # 策略 5(a): 优化器选择 ('adam', 'adamw', 'sgd')
        'optimizer_type': 'sgd',
        
        # 策略 4(b): 损失函数 ('ce', 'ce_smoothing') 与正则化项权重 (Weight Decay)
        'weight_decay': 0.0001,  # L2 正则化项权重
        'loss_type': 'ce',
        'label_smoothing': 0.1  # 仅在 ce_smoothing 时生效
    }

    train_model(experiment_config)
