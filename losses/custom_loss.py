from losses import SoftJaccardLoss, SoftDiceLoss, OhemCELoss
import torch.nn as nn
import torch


class CustomLoss:
    """
    Custom loss function.Implement your custom loss here
    """
    def __init__(self, input_size, ignore_index, reduction='mean'):
        b, c, h, w = input_size
        self.jaccard_loss = SoftJaccardLoss()
        self.dice_loss = SoftDiceLoss()
        # self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.ohemce_loss = OhemCELoss(threshold=0.7, n_min=b*h*w // 16, ignore_index=ignore_index, reduction=reduction)

    def train_loss(self, pred: torch.Tensor, pred8: torch.Tensor, pred16: torch.Tensor, target: torch.Tensor):
        """
        Train loss function
        :param pred: main output from the model, logits at each pixel (between -\infty and +\infty)
        :param pred8: low_feature maps x8 upsampled to the target size, logits at each pixel (between -\infty and +\infty)
        :param pred16: low_feature maps x16 upsampled to the target size, logits at each pixel (between -\infty and +\infty)
        :param target: ground truth labels (between 0 and C - 1)
        :return:
        Shape:
        - pred: torch.Tensor(B,C,W,H)
        - pred8: torch.Tensor(B,C,W,H)
        - pred16: torch.Tensor(B,C,W,H)
        - target: torch.Tensor(B,W,H)
        """
        loss_1 = self.ohemce_loss(pred, target.long())
        loss_2 = self.ohemce_loss(pred8, target.long())
        loss_3 = self.ohemce_loss(pred16, target.long())
        loss = loss_1 + loss_2 + loss_3
        return loss

    def val_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Validation loss function
        :param pred: main output from the model, logits at each pixel (between -\infty and +\infty)
        :param target: ground truth labels (between 0 and C - 1)
        Shape:
        - input: torch.Tensor(B,C,W,H)
        - target: torch.Tensor(B,W,H)
        """
        loss_1 = self.ohemce_loss(pred, target.long())
        loss = loss_1
        return loss
