import os
import torch
import torch.nn as nn

# 导入自定义的数据加载器与网络结构
from data.loaders import get_cifar_loader
from models.custom_cnn import CIFAR10ResidualNet

from utils.eval import eval
from utils.training import get_criterion


def test_model(model_path):
    # 1. 硬件设备准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载 Checkpoint 并提取元数据
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No checkpoint found at specified path: '{model_path}'")

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    metadata = checkpoint.get('metadata', {})
    config = metadata.get('config', {})
    
    if not config:
        raise ValueError("Error: 'config' field not found in checkpoint metadata. Cannot reconstruct the model architecture.")

    print(f"Successfully loaded metadata from experiment run.")

    # 3. 数据加载器初始化
    batch_size = config.get('batch_size', 128)
    test_loader = get_cifar_loader(
        batch_size=batch_size, 
        train=False, 
        shuffle=False, 
        num_workers=4
    )
    print(f"Test dataset loader initialized with batch size: {batch_size}")

    # 4. 依据配置动态实例化模型
    # 显式传入 config 中的 num_blocks 参数以动态配置网络深度，并设定默认值 [2, 2, 2] 确保向下兼容
    model = CIFAR10ResidualNet(
        base_channels=config['base_channels'],
        num_blocks=config.get('num_blocks', [1, 1, 1]),
        dropout_rate=config['dropout_rate'],
        activation_type=config['activation_type']
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    print("Model weights loaded into the network architecture.")

    # 5. 损失函数配置
    criterion = get_criterion(config['loss_type'], config.get('label_smoothing', 0.0))

    # 6. 执行独立测试集评估
    print("Evaluating on the standard CIFAR-10 test set...")
    test_loss, test_acc = eval(model, test_loader, criterion, device)

    # 7. 打印测试报告
    print("\n" + "=" * 60)
    print("                  CIFAR-10 TASK 1 TEST REPORT               ")
    print("=" * 60)
    print(f"Experiment Run Name:  {os.path.basename(os.path.dirname(model_path))}")
    print(f"Timestamp:            {metadata.get('timestamp', 'N/A')}")
    print(f"Saved at Epoch:       {metadata.get('epoch', 'N/A')}")
    print(f"Best Val Accuracy:    {metadata.get('best_val_accuracy', 0.0) * 100:.2f}%")
    print(f"Total Parameters:     {metadata.get('total_parameters', 0):,}")
    print("-" * 60)
    print(f"--> Final Test Loss:       {test_loss:.4f}")
    print(f"--> Final Test Accuracy:   {test_acc * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 指定需要评估的 best_model.pt 的路径
    MODEL_WEIGHT_PATH = 'runs/sgd_ce_smoothing_0.05_relu_ch64_b333_20260605_140540_epoch_100/best_model.pt'

    test_model(MODEL_WEIGHT_PATH)
