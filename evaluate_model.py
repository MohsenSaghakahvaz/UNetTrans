# evaluate_model.py
import os
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from model import SwinUNet
from dataset import SegmentationDataset
from metrics import pixel_accuracy, mean_iou  # متریک‌ها از قبل
from color_mapping import class2color  # عکس colormap: class index → رنگ RGB

def class_map_to_color(mask, class2color):
    """
    mask: [H, W] tensor with class indices
    return: [H, W, 3] numpy array (uint8 RGB)
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in class2color.items():
        color_mask[mask == cls] = color

    return color_mask

def evaluate_model(
    image_dir,
    mask_dir,
    color2class,
    class2color,
    num_classes=3,
    batch_size=1,
    save_dir="predictions"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Load Dataset
    test_dataset = SegmentationDataset(image_dir, mask_dir, color2class)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model = SwinUNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    total_acc = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)  # [B, C, H, W]
            pred_mask = torch.argmax(preds, dim=1)  # [B, H, W]

            # محاسبه متریک‌ها
            acc = pixel_accuracy(preds, masks)
            iou = mean_iou(preds, masks, num_classes)

            total_acc += acc.item()
            total_iou += iou.item()

            # ذخیره تصویر رنگی
            for i in range(images.size(0)):
                color_pred = class_map_to_color(pred_mask[i].cpu(), class2color)
                out_path = os.path.join(save_dir, f"pred_{idx*batch_size + i}.png")
                Image.fromarray(color_pred).save(out_path)
                

                # 🎨 اضافه کردن ماسک روی تصویر اصلی
                original_img = images[i]
                overlay = overlay_mask_on_image(original_img, color_pred)
                out_path_overlay = os.path.join(save_dir, f"overlay_{idx*batch_size + i}.png")
                Image.fromarray(overlay).save(out_path_overlay)
    

    avg_acc = total_acc / len(test_loader)
    avg_iou = total_iou / len(test_loader)

    print(f"✅ Evaluation Finished: Pixel Accuracy: {avg_acc:.4f} | mIoU: {avg_iou:.4f}")
