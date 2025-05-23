import torch.nn as nn
from patch_embedding import PatchEmbedding
from encoder_stage import EncoderStage
from swin_block import SwinBlock

class SwinEncoder(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels=3, embed_dim=96)

        # Stage 1: فقط SwinBlock بدون PatchMerging
        #self.stage1 = nn.Sequential(*[SwinBlock(dim=96) for _ in range(2)])
        self.stage1 = nn.Sequential(*[SwinBlock(dim=96, window_size=7) for _ in range(2)]) # استفاده از window-based self attention

        # Stage 2 to 4
        self.stage2 = EncoderStage(input_dim=96, output_dim=192, depth=2)
        self.stage3 = EncoderStage(input_dim=192, output_dim=384, depth=6)
        self.stage4 = EncoderStage(input_dim=384, output_dim=768, depth=2)

    def forward(self, x):
        skips = []  # برای skip connection

        x = self.patch_embed(x)         # (B, 96, 64, 64)
        x = self.stage1(x)              # (B, 96, 64, 64)
        skips.append(x)

        x = self.stage2(x)              # (B, 192, 32, 32)
        skips.append(x)

        x = self.stage3(x)              # (B, 384, 16, 16)
        skips.append(x)

        x = self.stage4(x)              # (B, 768, 8, 8)

        return x, skips  # x: feature نهایی، skips: خروجی هر stage
