from torch import nn
import torch.optim as optim


def select_loss_function(key: int, gamma: float = 2, alpha: float = 0.25,
                         smoothing: float = 0.1, neg_pos_ratio: float = 3.0) -> nn.Module:
    """Selects a loss function based on a given key.

    :param key: An integer value representing the desired loss function.
                0: CrossEntropyLoss
                1: FocalLoss
                2: MultiLabelMarginLoss
                3: LabelSmoothingLoss
    :param gamma: (optional) The gamma value for FocalLoss.
    :param alpha: (optional) The alpha value for FocalLoss.
    :param smoothing: (optional) The smoothing value for LabelSmoothingLoss.
    :param neg_pos_ratio: (optional) This parameter control the negatives bbox founded.
    :return: A loss function object.
    :raises: ValueError if the key is not supported.
    """
    if key == 0:
        return nn.CrossEntropyLoss()
    elif key == 1:
        return nn.FocalLoss(gamma=gamma, alpha=alpha)
    elif key == 2:
        return nn.MultiLabelMarginLoss()
    elif key == 3:
        return nn.LabelSmoothingLoss(smoothing=smoothing)
    else:
        raise ValueError(f"Unsupported loss function key: {key}")


def select_optimizer(key: int, model: nn.Module, lr: float = 0.001, momentum: float = 0.9,
                     weight_decay: float = 1e-4, nesterov: bool = True) -> optim.Optimizer:
    """Selects an optimizer based on a given key.

    :param key: An integer value representing the desired optimizer.
                0: Adam optimizer
                1: SGD optimizer
                2: Adagrad optimizer
                3: Adadelta optimizer
    :param model: The PyTorch model to be optimized.
    :param lr: The learning rate for the optimizer (default=0.001).
    :param weight_decay: The weight decay for the optimizer (default=1e-4).
    :return: An optimizer object.
    :raises: ValueError if the key is not supported.
    """
    if key == 0:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif key == 1:
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif key == 2:
        return optim.Adagrad(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif key == 3:
        return optim.Adadelta(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                              nesterov=nesterov)
    else:
        raise ValueError(f"Unsupported optimizer key: {key}")
