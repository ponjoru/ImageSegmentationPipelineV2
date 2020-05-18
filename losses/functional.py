import torch
import torch.nn.functional as F


def soft_dice_index(input: torch.Tensor, target: torch.Tensor, smooth=1e-4):
    """

    :param input: NxCxHxW
    :param target: NxHxW
    :return: tensor with shape C
    """
    b, c, w, h = input.size()

    dice_index = torch.zeros(c, dtype=torch.float, device=input.device)

    for class_index in range(c):
        if c == 1:
            tmp_target = (target == class_index + 1)
        else:
            tmp_target = (target == class_index)
        output = input[:, class_index, ...]

        num_preds = tmp_target.long().sum()

        if num_preds == 0:
            dice_index[class_index] = 0
        else:
            tmp_target = tmp_target.float()
            numerator = (output * tmp_target).sum()
            denominator = output.sum() + tmp_target.sum()
            dice_index[class_index] = 2 * (numerator + smooth) / (denominator + smooth)
    return dice_index


def soft_jaccard_index(input: torch.Tensor, target: torch.Tensor, smooth=1e-8):
    """

    :param input: NxCxHxW
    :param target: NxHxW
    :return: tensor with shape C
    """
    b, c, w, h = input.size()

    jaccard_index = torch.zeros(c, dtype=torch.float, device=input.device)
    for class_index in range(c):
        if c == 1:
            tmp_target = (target == class_index + 1)
        else:
            tmp_target = (target == class_index)
        output = input[:, class_index, ...]

        num_preds = tmp_target.long().sum()

        if num_preds == 0:
            jaccard_index[class_index] = 0
        else:
            tmp_target = tmp_target.float()
            intersection = (output * tmp_target).sum()
            union = output.sum() + tmp_target.sum() - intersection
            iou = (intersection + smooth) / (union + smooth)
            jaccard_index[class_index] = iou
    return jaccard_index


def focal_loss(input: torch.Tensor, target: torch.Tensor, threshold=0.5, alpha=None, gamma=2, class_weight=None):
    """
    Focal binary segmentation loss (https://arxiv.org/abs/1708.02002)
    :param input: Tensor(B,1,H,W)
    :param target: Tensor(B,H,W)
    :param threshold: float threshold (default: 0.5)
    :param alpha: weight of positive examples (default: None)
    :param gamma: int, gamma parameter of loss function (default: 2)
    :return: scalar
    """
    input = input.squeeze(dim=1)
    b, c, w, h = input.size()
    if c == 1:
        logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=class_weight)
    else:
        logpt = -F.cross_entropy(input, target, reduction="none", weight=class_weight)
    pt = torch.exp(logpt)
    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = -focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    return loss


def binary_ohem_crossentropy_loss(input: torch.Tensor, target: torch.Tensor, threshold, n_min, class_weight=None):
    input = input.squeeze(dim=1)
    b, c, w, h = input.size()
    if c == 1:
        loss = F.binary_cross_entropy_with_logits(input.float(), target, reduction="none", pos_weight=class_weight).view(-1)
    else:
        loss = F.cross_entropy(input, target, reduction="none", weight=class_weight)
    loss, _ = torch.sort(loss, descending=True)

    if loss[n_min] > threshold:
        loss = loss[loss > threshold]
    else:
        loss = loss[:n_min]

    return loss
