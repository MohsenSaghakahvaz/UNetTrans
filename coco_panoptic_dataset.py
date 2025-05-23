import json
import os
from PIL import Image
from torchvision.datasets import VisionDataset

class CocoPanoptic(VisionDataset):
  def __init__(self, root, annFile, mask_root, transform=None):
    super().__init__(root, transform=transform)
    self.root = root
    self.transform = transform
    self.mask_root = mask_root
    # بارگذاری annotations
    with open(annFile, 'r') as f:
      self.annotations = json.load(f)

    # لیست فایل‌های تصاویر و ماسک‌ها
    self.images = {ann['id']: ann for ann in self.annotations['images']}
    self.image_ids = list(self.images.keys())
    self.annotations = {ann['image_id']: ann for ann in self.annotations['annotations']}

  def __getitem__(self, index):
    image_id = self.image_ids[index]
    image_info = self.annotations[image_id]

    # بارگذاری تصویر
    img = self.images[image_id]
    img_path = os.path.join(self.root, img['file_name'])
    image = Image.open(img_path).convert("RGB")

    # بارگذاری ماسک
    mask_path = os.path.join(self.mask_root, image_info['file_name'])
    mask = Image.open(mask_path)

    if self.transform:
      image = self.transform(image)
      mask = self.transform(mask)

    return image, mask

  def __len__(self):
    return len(self.image_ids)
