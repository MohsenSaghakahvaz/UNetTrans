import torch
import torch.nn as nn
import torch.nn.init as init

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # مقداردهی وزن‌ها به صورت ثابت
        init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # تبدیل به توکن‌ها
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        return x