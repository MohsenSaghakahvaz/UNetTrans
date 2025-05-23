import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mlp import MLP
from window_attention import WindowAttention, window_partition, window_reverse

class SwinBlock(nn.Module):
    def __init__(self, dim, window_size = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        #self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.attn = WindowAttention(dim, window_size) # برای دقت بیشتر
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        
        self.window_size = window_size

#    def forward(self, x):
#        B, C, H, W = x.shape
#        x = rearrange(x, 'b c h w -> b (h w) c')  # flatten spatial dims
#        shortcut = x
#        x = self.norm1(x)
#        x, _ = self.attn(x, x, x)
#        x = shortcut + x

#        x = x + self.mlp(self.norm2(x))
#        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
#        return x

    def forward(self, x):
        B, C, H, W = x.shape

        shortcut = x
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # پد کردن تصویر
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        Hp, Wp = x.shape[2], x.shape[3]

        # Partition
        x_windows = window_partition(x, self.window_size)  # (B*nW, win, win, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        # Reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Remove padding
        x = x[:, :, :H, :W]

        x = x + shortcut

        x_ = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.mlp(x)
        x = x + x_

        return x
