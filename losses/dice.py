import torch
import torch.nn as nn
import losses.functional as F

# TODO: support classes feature


class SoftDiceLoss(nn.Module):
    """
    Soft dice loss function (a.k.a f1 score loss, 1 - 2*|X^Y| / |X|+|Y|) (https://arxiv.org/abs/1707.03237)
    :param from_logits: whether input is logit(between -\infty and +\infty) or
                        class probabilities at each prediction (between 0 and 1) (default: True)
    :param reduction: str - reduction mode, one of 'mean', 'sum', 'none', (default: 'mean') (if 'none' returns per class
                      dice loss)
    """
    def __init__(self, from_logits=True, reduction='mean'):

        super(SoftDiceLoss, self).__init__()
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: NxCxHxW
        :param target: NxHxW
        :return: scalar
        """
        b, c, w, h = input.size()

        if self.from_logits:
            if c > 1:
                input = torch.softmax(input, dim=1)
            else:
                input = torch.sigmoid(input)

        smooth = 1e-4

        dice_loss = 1 - F.soft_dice_index(input, target, smooth)

        if self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss.mean()
