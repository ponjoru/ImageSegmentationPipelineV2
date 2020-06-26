from losses import SoftJaccardLoss, SoftDiceLoss, OhemCELoss
import torch.nn as nn
import torch


class CustomLoss:
    """
    Custom loss function.Implement your custom loss here
    """
    def __init__(self, input_size, ignore_index, reduction='mean', delay=None):
        b, c, h, w = input_size
        self.jaccard_loss = SoftJaccardLoss()
        self.dice_loss = SoftDiceLoss()
        # self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.ce_loss = OhemCELoss(threshold=0.7, n_min=b*h*w // 16, ignore_index=ignore_index, reduction=reduction)
        self.delay = delay

    def train_loss(self, input: torch.Tensor, target: torch.Tensor, epoch: int):
        """
        Train loss function
        :param input: main output from the model, logits at each pixel (between -\infty and +\infty)
        :param in16: low_feature maps x16 upsampled to the target size, logits at each pixel (between -\infty and +\infty)
        :param in32: low_feature maps x32 upsampled to the target size, logits at each pixel (between -\infty and +\infty)
        :param target: ground truth labels (between 0 and C - 1)
        :return:
        Shape:
        - input: torch.Tensor(B,C,W,H)
        - in16: torch.Tensor(B,C,W,H)
        - in32: torch.Tensor(B,C,W,H)
        - target: torch.Tensor(B,W,H)
        """
        loss_1 = self.ce_loss(input, target.long())
        if self.delay and epoch > self.delay:
            loss_2 = self.jaccard_loss(input, target)
            loss_3 = self.dice_loss(input, target)
            loss = 0.8 * loss_1 + loss_2 + loss_3
        else:
            loss = loss_1
        return loss

    def val_loss(self, input: torch.Tensor, target: torch.Tensor, epoch: int):
        """
        Validation loss function
        :param input: main output from the model, logits at each pixel (between -\infty and +\infty)
        :param target:
        :param target: ground truth labels (between 0 and C - 1)
        Shape:
        - input: torch.Tensor(B,C,W,H)
        - target: torch.Tensor(B,W,H)
        """
        loss1 = self.ce_loss(input, target.long())
        return loss1