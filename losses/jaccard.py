import torch
import torch.nn as nn
import losses.functional as F

# TODO: support classes feature


class SoftJaccardLoss(nn.Module):
    """
    Soft dice loss function (a.k.a f1 score loss, 1 - 2*|X^Y| / |X|+|Y|)
    :param from_logits: whether input is logit(between -\infty and +\infty) or
                        class probabilities at each prediction (between 0 and 1) (default: True)
    :param reduction: one of 'mean', 'sum', 'none', (default: 'mean') (if 'none' returns per class jaccard loss)
    """
    def __init__(self, from_logits=True, reduction='mean'):
        super(SoftJaccardLoss, self).__init__()
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """

        :param input: [NxCxHxW]
        :param target: [NxHxW]
        :return scalar
        """
        b, c, w, h = input.size()

        if self.from_logits:
            if c > 1:
                input = torch.softmax(input, dim=1)
            else:
                input = torch.sigmoid(input)

        smooth = 1e-4

        iou_loss = 1 - F.soft_jaccard_index(input, target, smooth)

        if self.reduction == 'sum':
            return iou_loss.sum()
        return iou_loss.mean()
