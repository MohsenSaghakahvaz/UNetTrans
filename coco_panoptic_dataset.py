import json
import os
from PIL import Image
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

class CocoPanoptic(VisionDataset):
    def __init__(self, root, annFile, mask_root, image_size =(256,256), transform=None):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        
        # Transform برای ماسک (فقط resize)
        self.mask_resize = transforms.Resize(image_size, interpolation=Image.NEAREST)
        
        self.mask_root = mask_root
        # بارگذاری annotations
        with open(annFile, 'r') as f:
            self.annotations = json.load(f)

        # لیست فایل‌های تصاویر و ماسک‌ها
        self.images = {ann['id']: ann for ann in self.annotations['images']}
        self.image_ids = list(self.images.keys())
        self.categories = self.annotations['categories']
        self.annotations = {ann['image_id']: ann for ann in self.annotations['annotations']}

        cmap = generate_colors([cat["id"] for cat in self.categories])
        self.color2class = cmap
    
        for idx, cat in enumerate(self.categories):
            cat["color"] = cmap[idx]["color"]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.annotations[image_id]

        # بارگذاری تصویر
        img = self.images[image_id]
        img_path = os.path.join(self.root, img['file_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # بارگذاری ماسک
        mask_path = os.path.join(self.mask_root, image_info['file_name'])
        mask = Image.open(mask_path).convert('RGB')
        mask = self.mask_resize(mask)
        # تبدیل ماسک RGB به کلاس
        mask = self.rgb_mask_to_class(np.array(mask), self.color2class)

        return image, mask

    def __len__(self):
        return len(self.image_ids)

    def rgb_mask_to_class(self, mask_np, color2class):
        """ تبدیل ماسک رنگی به class index map """
        class_map = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        for obj in color2class:
            class_id = obj["id"]
            color = obj["color"]
            
            r, g, b = color
            match = (mask_np[:, :, 0] == r) & (mask_np[:, :, 1] == g) & (mask_np[:, :, 2] == b)
            class_map[match] = class_id
        return torch.from_numpy(class_map)
    
def generate_colors(ids):
    # از cmap نوع hsv استفاده می‌کنیم که در بازهٔ 0 تا 1 رنگ‌های مختلف دارد
    cmap = plt.get_cmap('hsv', len(ids))  # count رنگ متمایز

    color_list = []
    for i in range(cmap.N):
        rgba = cmap(i)  # مقداری شبیه (R, G, B, A) بر می‌گرداند، در بازه 0..1
        rgb_255 = tuple(int(c * 255) for c in rgba[:3])  # تبدیل به بازه 0..255
        color_list.append({"id": ids[i], "color": rgb_255})

    return color_list