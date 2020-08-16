import torch
import torch.nn as nn
import losses.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, threshold=0.5, reduction="mean"):
        """
        Focal balanced binary segmentation loss (https://arxiv.org/abs/1708.02002).
        :param alpha: weight of positive examples (default: 0.25)
        :param gamma: int, gamma parameter of loss function (default: 2)
        :param threshold: float threshold (default: 0.5)
        :param reduction: str - reduction mode, one of 'mean', 'sum', (default: 'mean')
        :return: scalar
        """
        super(BinaryFocalLoss, self).__init__()
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
        loss = F.binary_focal_loss(input, target, self.threshold, self.alpha, self.gamma)
        if self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss


class MulticlassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, threshold=0.5, reduction="mean"):
        """
        Focal balanced multiclass segmentation loss (https://arxiv.org/abs/1708.02002).
        Converts target to one_hot and Sums up binary focal losses over each classes with the corresponding alpha
        weights. Slower than FocalLoss. Better use for multiclass segmentation with known prior class weights.
        :param threshold: float threshold (default: 0.5)
        :param alpha: list or np.array of C items: weights of positive examples (default: None)
        :param gamma: int, gamma parameter of loss function (default: 2)
        :param reduction: str - reduction mode, one of 'mean', 'sum', (default: 'mean')
        :return: scalar
        """
        super(MulticlassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: Tensor(B,C,H,W)
        :param target: Tensor(B,H,W)
        :return:
        """
        loss = F.focal_loss(input, target, self.threshold, self.alpha, self.gamma)
        if self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss


class MulticlassFocalLossV2(nn.Module):
    def __init__(self, gamma=2, threshold=0.5, reduction="mean"):
        """
        Focal unbalanced multiclass segmentation loss (https://arxiv.org/abs/1708.02002).
        Based on nn.CrossEntropy loss without one-hot encoding convertation. Faster than MulticlassFocalLoss
        Better use for multiclass segmentation with unknown prior class weights.
        :param threshold: float threshold (default: 0.5)
        :param gamma: int, gamma parameter of loss function (default: 2)
        :param reduction: str - reduction mode, one of 'mean', 'sum', (default: 'mean')
        :return: scalar
        """
        super(MulticlassFocalLossV2, self).__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: Tensor(B,C,H,W)
        :param target: Tensor(B,H,W)
        :return:
        """
        loss = F.focal_loss_v2(input, target, self.threshold, self.gamma)
        if self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss