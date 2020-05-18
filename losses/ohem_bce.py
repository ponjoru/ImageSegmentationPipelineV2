import torch
import torch.nn as nn
import losses.functional as F


class OhemCELoss(nn.Module):
    """
    Implements online hard extreme mining binary cross entropy with logits loss
    :param threshold: threshold
    :param n_min: top n_min negative values above threshold of cross entropy output to impact the total loss
    :param cuda: whether to use cuda or not (default: True)
    :param class_weight: positive class weight (default: None)
    :param reduction: str: 'mean' or 'sum'. (default: 'mean')
    """
    def __init__(self, threshold, n_min, cuda=True, class_weight=None, reduction="mean"):
        super(OhemCELoss, self).__init__()
        self.threshold = -torch.log(torch.tensor(threshold, dtype=torch.float))
        if cuda:
            self.threshold = self.threshold.cuda()
        self.n_min = n_min
        self.class_weight = class_weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: Tensor of shape (B,1,H,W)
        :param target: Tensor of shape (B,H,W)
        :return: scalar
        """
        loss = F.binary_ohem_crossentropy_loss(input, target, self.threshold, self.n_min, self.class_weight)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


