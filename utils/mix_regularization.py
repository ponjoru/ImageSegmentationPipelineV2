import torch
import random
import numpy as np


def mixup(data, target, alpha=1.0):
    gamma = np.random.beta(alpha, alpha)
    gamma = max(gamma, (1 - gamma))
    with torch.no_grad():
        perm = torch.randperm(data.size(0))
        perm_data = data[perm]
        perm_target = target[perm]
        data = gamma * data + (1 - gamma) * perm_data
    return data, target, perm_target, gamma


def mix_criterion(criterion, pred, tgt_a, tgt_b, gamma):
    return gamma * criterion(**pred, target=tgt_a) + (1 - gamma) * criterion(**pred, target=tgt_b)


def cutmix(data, target, alpha=1.0):
    gamma = np.random.beta(alpha, alpha)
    gamma = max(gamma, (1 - gamma))
    with torch.no_grad():
        perm = torch.randperm(data.size()[0]).cuda()
        perm_target = target[perm]
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), gamma)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[perm, :, bbx1:bbx2, bby1:bby2]
        # adjust gamma to exactly match pixel ratio
        gamma = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, target, perm_target, gamma


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def random_joint_mix(data, target, CutMix=True, MixUp=True, p=0.5):
    if not MixUp and not CutMix:
        return data, target, None, None
    if CutMix and MixUp:
        p_star = random.random()
        if p_star >= p:
            return cutmix(data, target)
        else:
            return mixup(data, target)
    if CutMix:
        return cutmix(data, target)
    else:
        return mixup(data, target)
