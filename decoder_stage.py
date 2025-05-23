import torch.nn as nn
from swin_block import SwinBlock

class DecoderStage(nn.Module):
    def __init__(self, input_dim, skip_dim, output_dim, depth=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # تطبیق channel ها
        self.input_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_dim, output_dim, kernel_size=1)

        # پردازش بعد از ترکیب
        self.blocks = nn.Sequential(*[SwinBlock(output_dim) for _ in range(depth)])

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.input_proj(x)
        skip = self.skip_proj(skip)

        # تنظیم اندازه در صورت اختلاف spatial
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = x + skip
        x = self.blocks(x)
        return x
