import torch.nn as nn
from decoder_stage import DecoderStage

class SwinDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DecoderStage(input_dim=768, skip_dim=384, output_dim=384, depth=2)
        self.stage2 = DecoderStage(input_dim=384, skip_dim=192, output_dim=192, depth=2)
        self.stage3 = DecoderStage(input_dim=192, skip_dim=96, output_dim=96, depth=2)
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.output_head = nn.Conv2d(96, 1, kernel_size=1)  # فرض بر binary segmentation

    def forward(self, x, skips):
        x = self.stage1(x, skips[2])  # 8x8 → 16x16
        x = self.stage2(x, skips[1])  # 16x16 → 32x32
        x = self.stage3(x, skips[0])  # 32x32 → 64x64
        x = self.final_upsample(x)    # 64x64 → 256x256
        x = self.output_head(x)       # خروجی نهایی: 256x256x1
        return x
