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


def binary_focal_loss(input: torch.Tensor, target: torch.Tensor, threshold=0.5, alpha=0.25, gamma=2):
    # input = input.squeeze(dim=1)
    logpt = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = focal_term * logpt

    if alpha:
        alpha = (alpha * target + (1 - alpha) * (1 - target))
        loss = loss * alpha
    return loss

def focal_loss(input: torch.Tensor, target: torch.Tensor, threshold=0.5, alpha=None, gamma=2):
    b, c, h, w = input.size()
    with torch.no_grad():
        one_hot = torch.zeros(input.size())
        one_hot = one_hot.to(input.device)
        one_hot.scatter_(dim=1, index=target.unsqueeze(1), value=1.0)
    loss = 0.0
    for cls in range(c):
        if alpha is not None:
            a = alpha[cls]
        else:
            a = alpha
        loss += binary_focal_loss(input[:,cls], one_hot[:,cls], alpha=a, gamma=gamma, threshold=threshold)
    return loss

def focal_loss_v2(input: torch.Tensor, target: torch.Tensor, threshold=0.5, gamma=2, class_weight=None):
    """
    Focal segmentation loss (https://arxiv.org/abs/1708.02002)
    :param input: Tensor(B,C,H,W)
    :param target: Tensor(B,H,W)
    :param threshold: float threshold (default: 0.5)
    :param alpha: weight of positive examples (default: None)
    :param gamma: int, gamma parameter of loss function (default: 2)
    :return: scalar
    """
    logpt = -F.cross_entropy(input, target, reduction="none", weight=class_weight)
    pt = torch.exp(logpt)
    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = -focal_term * logpt

    return loss


def ohem_crossentropy_loss(input: torch.Tensor, target: torch.Tensor, threshold, n_min, ignore_index, class_weight=None):
    b, c, w, h = input.size()
    if c == 1:
        target = target.unsqueeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=class_weight).view(-1)
    else:
        loss = F.cross_entropy(input, target, reduction="none", weight=class_weight, ignore_index=ignore_index).view(-1)
    loss, _ = torch.sort(loss, descending=True)

    if loss[n_min] > threshold:
        loss = loss[loss > threshold]
    else:
        loss = loss[:n_min]

    return loss


if __name__ == '__main__':
    # input = torch.randn((4, 19, 1024, 2048))
    # target = (torch.randn((4, 1024, 2048)).abs().long())

    input = torch.sigmoid(torch.randn((4, 1, 1024, 2048)))
    target = torch.sigmoid(torch.randn((4, 1024, 2048)))

    l = ohem_crossentropy_loss(input, target, threshold=0.7, n_min=4 * 1024 * 2048 // 16)
    l1 = focal_loss(input, target)
    k = 5