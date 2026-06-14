import time
import torch
from tqdm import tqdm
from thop import profile

from data.loaders import get_cifar_loader
from models.custom_cnn import CIFAR10ResidualNet, get_number_of_parameters
from models.vgg import VGG_A_BatchNorm  # 导入新增的 VGG 模型
from utils.eval import eval
from utils.training import get_criterion, get_optimizer


def benchmark_single_model(model_name, model_builder, config, train_loader, device):
    """
    对单个模型进行完整的理论计算量评估与训练吞吐量测试
    """
    print("\n" + "="*60)
    print(f" 正在评测模型: {model_name} ")
    print("="*60)
    
    # ----------------------------------------------------
    # STEP 1: 计算理论计算量 (FLOPs) 与参数量
    # ----------------------------------------------------
    print(f"\n[阶段 1] 正在评估 {model_name} 的理论计算量与参数量...")
    
    # 实例化一个临时模型用于 Profile
    model_eval = model_builder().to(device)
    model_eval.eval()

    # 调用项目自带的函数统计参数量
    custom_params = get_number_of_parameters(model_eval)

    if profile is not None:
        dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-10 尺寸
        macs, thop_params = profile(model_eval, inputs=(dummy_input,), verbose=False)
        flops = 2 * macs
        print(f"  -> 单张图像前向推理计算量 (FLOPs): {flops:,} ({flops / 1e6:.2f} MFLOPs)")
        print(f"  -> [get_number_of_parameters 统计] 总参数量: {custom_params:,}")
    else:
        print("  -> [警告] 因缺少 thop 库，未能计算 FLOPs。")
        print(f"  -> [get_number_of_parameters 统计] 总参数量: {custom_params:,}")
    
    del model_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # STEP 2: 测试训练吞吐量 (Training Throughput)
    # ----------------------------------------------------
    print(f"\n[阶段 2] 正在评估 {model_name} 的训练吞吐量 (Warmup vs Stable)...")
    
    model = model_builder().to(device)
    criterion = get_criterion(config['loss_type'], config.get('label_smoothing', 0.0))
    optimizer = get_optimizer(model, config)
    
    stable_throughput = 0.0

    # 仅跑 2 个 Epoch 捕获稳态
    for epoch in range(1, 3):
        model.train()
        total_samples = 0
        start_time = time.time()
        
        for X, y in tqdm(train_loader, desc=f"  {model_name} - Epoch {epoch}/2"):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_samples += X.size(0)
            
        epoch_time = time.time() - start_time
        throughput = total_samples / epoch_time
        
        if epoch == 1:
            print(f"     [Warmup 轮] Epoch 1 训练吞吐量: {throughput:.1f} img/s | 耗时: {epoch_time:.2f} 秒")
        elif epoch == 2:
            stable_throughput = throughput
            print(f"     [稳态基准] Epoch 2 训练吞吐量: {throughput:.1f} img/s | 耗时: {epoch_time:.2f} 秒")

    # 清理显存避免影响下一个模型的评测
    del model, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return custom_params, stable_throughput


def run_benchmark_suite(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试硬件环境 (Device): {device}")
    
    # 统一初始化数据加载器，避免重复加载
    print("\n正在初始化 CIFAR-10 数据加载器...")
    train_loader = get_cifar_loader(
        batch_size=config['batch_size'], 
        train=True, 
        val_split=config.get('val_split', 0.1), 
        is_val=False, 
        shuffle=True, 
        num_workers=4
    )

    # 定义需要评测的模型列表及其构造方式
    models_to_benchmark = [
        {
            "name": "CIFAR10ResidualNet",
            "builder": lambda: CIFAR10ResidualNet(
                base_channels=config['base_channels'],
                num_blocks=config.get('num_blocks', [2, 2, 2]),
                dropout_rate=config['dropout_rate'],
                activation_type=config['activation_type']
            )
        },
        {
            "name": "VGG_A_BatchNorm",
            "builder": lambda: VGG_A_BatchNorm(inp_ch=3, num_classes=10)
        }
    ]

    results = {}
    for item in models_to_benchmark:
        params, throughput = benchmark_single_model(
            model_name=item["name"],
            model_builder=item["builder"],
            config=config,
            train_loader=train_loader,
            device=device
        )
        results[item["name"]] = {"Params": params, "Throughput": throughput}

    # ====================================================
    # 打印最终对比摘要表
    # ====================================================
    print("\n" + "*"*60)
    print(" " * 20 + "基准测试最终对比摘要")
    print("*"*60)
    print(f"{'Model Name':<30} | {'Total Params':<15} | {'Stable Throughput':<20}")
    print("-"*65)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Params']:<15,} | {metrics['Throughput']:>14.1f} img/s")
    print("*"*60 + "\n")


if __name__ == "__main__":
    experiment_config = {
        'epochs': 2,  
        'batch_size': 128,
        'lr': 0.01,
        'val_split': 0.1,
        'base_channels': 64,
        'num_blocks': [3, 3, 3],
        'dropout_rate': 0.3,
        'activation_type': 'gelu',
        'optimizer_type': 'sgd',
        'weight_decay': 0.0001,
        'loss_type': 'ce',
        'label_smoothing': 0.1
    }
    
    run_benchmark_suite(experiment_config)
