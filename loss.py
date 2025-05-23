import torch
import torch.nn.functional as F

def combined_loss(pred, target, alpha=0.5, beta=0.5):
    """
    alpha: weight for CrossEntropy
    beta: weight for Dice (or IoU)
    """
    crossEntropy = F.cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    #IoU = iou_loss(pref, target, num_classes)
    return alpha * crossEntropy + beta * dice
    #return alpha * crossEntropy + beta * dice + 0.2 * IoU

def iou_loss(pred, target, smooth=1e-6):
    """
    pred: [B, C, H, W] — logits
    target: [B, H, W] — class indices (int)
    """
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = (pred + target_onehot - pred * target_onehot).sum(dim=(2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()

def dice_loss(pred, target, smooth=1e-6):
    """
    pred: [B, C, H, W] — logits
    target: [B, H, W] — class indices (int)
    """
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()