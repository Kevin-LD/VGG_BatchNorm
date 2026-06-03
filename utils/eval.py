import torch

def eval(model, loader, criterion, device):
    """
    模型评估函数 (验证/测试)
    计算并返回指定数据集上的 Loss 和 Accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total

    return epoch_loss, accuracy
