import torch
from src.utils.optimizer.radam import RAdam
from tqdm import tqdm


_optimizer_factory = {"adam": torch.optim.Adam, "radam": RAdam, "sgd": torch.optim.SGD}


def make_optimizer(cfg, net, stage = 'distillation'):
    if stage == 'fine-tuning':
        # 如果是微调阶段，优先使用 ft_lr ，如果未定义则使用 lr
        lr = cfg.train.get('ft_lr', cfg.train.lr)
    else:
        lr = cfg.train.lr

    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    # 直接获取所有需要梯度的参数，无需循环！
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())

    if "adam" in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](
            trainable_params, lr, weight_decay=weight_decay, eps=eps
        )
    else:
        optimizer = _optimizer_factory[cfg.train.optim](trainable_params, lr, momentum=0.9)

    return optimizer
