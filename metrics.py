import torch

def pixel_accuracy(preds, targets):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()


def mean_iou(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)  # shape: [B, H, W]
    ious = []

    preds = preds.to(targets.device)  # 👈 مطمئن شو روی یک device هستن

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(torch.tensor(1.0, device=targets.device))  # 👈 مهم: روی همون device
        else:
            ious.append(intersection / union)

    return torch.mean(torch.stack(ious))