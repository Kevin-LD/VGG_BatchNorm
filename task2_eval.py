import os
import torch
import torch.nn as nn

# 导入自定义的数据加载器与 VGG 网络结构
from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm

from utils.eval import eval

def test_vgg_model(model_path):
    # 1. 硬件设备准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载 Checkpoint 并提取元数据
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No checkpoint found at specified path: '{model_path}'")

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    metadata = checkpoint.get('metadata', {})
    model_name = metadata.get('model_name', None)
    
    if not model_name:
        raise ValueError("Error: 'model_name' field not found in checkpoint metadata. Cannot reconstruct VGG architecture.")

    print(f"Successfully loaded metadata from experiment run.")

    # 3. 数据加载器初始化
    test_loader = get_cifar_loader(train=False, augment=False)
    print("Test dataset loader initialized.")

    # 4. 依据配置动态实例化 VGG 模型
    if model_name == "VGG_A":
        model = VGG_A()
    elif model_name == "VGG_A_BatchNorm":
        model = VGG_A_BatchNorm()
    else:
        raise ValueError(f"Unknown model_name detected in checkpoint: '{model_name}'")
        
    model = model.to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    print("Model weights loaded into the network architecture.")

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 5. 损失函数配置（Task 2 统一采用标准交叉熵）
    criterion = nn.CrossEntropyLoss()

    # 6. 执行独立测试集评估
    print(f"Evaluating on the standard CIFAR-10 test set...")
    test_loss, test_acc = eval(model, test_loader, criterion, device)

    # 7. 打印测试报告
    print("\n" + "=" * 60)
    print("                  CIFAR-10 TASK 2 TEST REPORT               ")
    print("=" * 60)
    print(f"Model Architecture:   {model_name}")
    print(f"Experiment Run Name:  {os.path.basename(os.path.dirname(model_path))}")
    print(f"Timestamp:            {metadata.get('save_time', 'N/A')}")
    print(f"Saved at Epoch:       {metadata.get('epoch', 'N/A')}")
    print(f"Best Val Accuracy:    {metadata.get('best_val_accuracy', 0.0) * 100:.2f}%")
    print(f"Total Parameters:     {total_params:,}")
    print("-" * 60)
    print(f"--> Final Test Loss:       {test_loss:.4f}")
    print(f"--> Final Test Accuracy:   {test_acc * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 指定需要评估的最佳模型的路径
    MODEL_WEIGHT_PATH = 'runs/task2/VGG_OriginalPaper_Landscape_20260608_115158/VGG_A_BatchNorm_OriginalMethod_lr_0.1/best_model.pt'

    test_vgg_model(MODEL_WEIGHT_PATH)
