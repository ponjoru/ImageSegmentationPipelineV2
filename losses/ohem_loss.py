import torch
import torch.nn as nn
import losses.functional as F


class OhemCELoss(nn.Module):
    """
    Implements Online Hard Example Mining cross entropy loss
    :param threshold: threshold
    :param n_min: top n_min negative values above threshold of cross entropy output to impact the total loss
    :param cuda: whether to use cuda or not (default: True)
    :param class_weight: positive class weight (default: None)
    :param reduction: str: 'mean' or 'sum'. (default: 'mean')
    """
    def __init__(self, threshold, n_min, ignore_index=255, cuda=True, class_weight=None, reduction="mean"):
        super(OhemCELoss, self).__init__()
        self.threshold = -torch.log(torch.tensor(threshold, dtype=torch.float))
        if cuda:
            self.threshold = self.threshold.cuda()
        self.n_min = n_min
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: Tensor of shape (B,C,H,W)
        :param target: Tensor of shape (B,H,W)
        :return: scalar
        """
        loss = F.ohem_crossentropy_loss(input, target, self.threshold, self.n_min, self.ignore_index, self.class_weight)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class OHEMLoss(nn.Module):
    """
    Implements Online Hard Example Mining extension for a given loss
    :param loss - instance of the loss function to apply OHEM extension to.
                  Note: it must have parameters: reduction='none'
    :param threshold: threshold
    :param n_min: top n_min negative values above threshold of the loss output to impact the total loss
    :param reduction: str: 'mean' or 'sum'. (default: 'mean')
    """

    def __init__(self, loss, threshold, n_min, ignore_index=255, reduction="mean"):
        super(OHEMLoss, self).__init__()
        self.criterion = loss
        self.threshold = -torch.log(torch.tensor(threshold, dtype=torch.float))
        self.n_min = n_min
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: Tensor of shape (B,C,H,W)
        :param target: Tensor of shape (B,H,W)
        :return: scalar
        """
        loss = self.criterion(input, target, reduction='none')
        loss, _ = torch.sort(loss, descending=True)

        if loss[self.n_min] > self.threshold:
            loss = loss[loss > self.threshold]
        else:
            loss = loss[:self.n_min]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
