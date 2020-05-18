import torch
import torch.nn as nn
import losses.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, threshold=0.5, reduction="mean"):
        """
        Focal binary segmentation loss (https://arxiv.org/abs/1708.02002)
        :param threshold: float threshold (default: 0.5)
        :param alpha: weight of positive examples (default: None)
        :param gamma: int, gamma parameter of loss function (default: 2)
        :param reduction: str - reduction mode, one of 'mean', 'sum', (default: 'mean')
        :return: scalar
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: Tensor(B,1,H,W)
        :param target: Tensor(B,H,W)
        :return:
        """
        loss = F.focal_loss(input, target, self.threshold, self.alpha, self.gamma)
        if self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss

