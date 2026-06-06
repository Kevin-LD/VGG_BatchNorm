import torch
import numpy as np
from tqdm import tqdm

def eval(model, loader, criterion, device, return_preds=False, desc="Evaluating"):
    """
    模型评估函数 (验证/测试)
    计算并返回指定数据集上的 Loss 和 Accuracy。
    如果 return_preds=True，则额外返回全量预测值与真实标签的 NumPy 数组。
    """
    # 1. 记录模型原有的训练/评估状态，以便后续恢复
    was_training = model.training

    # 2. 开启评估模式 (确保 Dropout 和 BatchNorm 行为正确)
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0

    # 3. 初始化用于收集全网预测结果的容器
    all_preds = [] if return_preds else None
    all_targets = [] if return_preds else None

    # 4. 引入 tqdm 进度条
    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # 5. 如果需要返回结果，将当前 Batch 的数据转移至 CPU 并展平追加至列表中
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

    pbar.close()

    # 6. 计算最终的平均损失与准确率
    epoch_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # 恢复模型进入评估函数前的原始状态
    if was_training:
        model.train()

    if return_preds:
        return epoch_loss, accuracy, np.array(all_preds), np.array(all_targets)
    
    return epoch_loss, accuracy
