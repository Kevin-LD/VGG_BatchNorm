from torch import nn
from torch import optim

def get_criterion(loss_type, label_smoothing=0.0):
    """
    不同的损失函数
    """
    loss_type = loss_type.lower()
    if loss_type == 'ce':
        # 标准交叉熵损失
        return nn.CrossEntropyLoss()
    elif loss_type == 'ce_smoothing':
        # 带有标签平滑的交叉熵（一种强正则化手段，防止模型过度自信，对应要求 4b）
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_optimizer(model, config):
    opt_type = config['optimizer_type'].lower()
    lr = config['lr']
    weight_decay = config['weight_decay'] # L2 正则化
    
    if opt_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")
