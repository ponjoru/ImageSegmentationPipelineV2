import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss function: multiclass and binary
    In multiclass case converts target to one-hot encoding, and modifies pixels:
        1 -> 1 - smoothing + smoothing / classes
        0 -> smoothing
        And then applies KLDiv loss onto prediction and smoothed one-hot target
    In binary case:
        modifies pixels the same way, and applies Binary Cross Entropy loss
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.KLDiv = nn.KLDivLoss()
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def smooth_one_hot(target, classes, smoothing):
        """
        Converts target to smoothed one-hot encoding
        :param target: tensor Nx1xHxW
        :param classes: number of classes
        :param smoothing: smoothing coefficient
        :return:
        """
        b, h, w = target.size()
        target = target.unsqueeze(1).long()
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        with torch.no_grad():
            one_hot = torch.empty((b, classes, h, w)).fill_(smoothing).cuda()
            one_hot.scatter_(dim=1, index=target, value=confidence + smoothing / classes)
        return one_hot

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Loss evaluation
        :param pred: tensor [BxCxHxW]
        :param target: tensor [Bx1XHxW]
        :return:
        """
        _, c, _, _ = pred.size()
        if c == 1:
            target_star = torch.abs(target - self.smoothing)
            return self.bce(pred, target_star.unsqueeze(1))
        pred = F.log_softmax(pred, dim=1)
        target_star = self.smooth_one_hot(target, classes=c, smoothing=self.smoothing)
        return self.KLDiv(pred, target_star)
