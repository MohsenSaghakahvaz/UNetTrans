import numpy as np
import torch
from coco_panoptic_dataset import CocoPanoptic
from torch.utils.data import DataLoader, Dataset, Subset
from loss import combined_loss
from metrics import pixel_accuracy, mean_iou


def train_model(
    model, 
    image_dir,
    mask_dir,
    ann_file,
    transform,
    num_classes=80,
    epochs=20,
    batch_size=4,
    lr=1e-4,
    train_size = 0.7,
    test_size = 0.3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    full_dataset = CocoPanoptic(
        root = image_dir,
        annFile = ann_file,
        mask_root = mask_dir,
        image_size = (256, 256),
        transform = transform)

    total_size = len(full_dataset)

    # Create Indics and shuffle
    indices = np.arange(total_size)
    np.random.seed(42)  # برای تکرارپذیری
    np.random.shuffle(indices)

    # Split indics
    split = int(0.7 * total_size)
    train_indices = indices[:split]
    test_indices = indices[split:]

    # Sub Datasets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler (اختیاری)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_iou = 0.0

        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            loss = combined_loss(preds, masks)
            acc = pixel_accuracy(preds, masks)
            iou = mean_iou(preds, masks, num_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            total_iou += iou.item()

        scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_acc / len(train_dataloader)
        avg_iou = total_iou / len(train_dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | mIoU: {avg_iou:.4f}")

        # ذخیره مدل با بهترین IoU
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved.")