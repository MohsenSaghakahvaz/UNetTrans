import torch.nn as nn
from patch_merging import PatchMerging
from swin_block import SwinBlock

class EncoderStage(nn.Module):
    def __init__(self, input_dim, output_dim, depth=2):
        super().__init__()
        self.patch_merging = PatchMerging(input_dim)
        self.blocks = nn.Sequential(*[SwinBlock(output_dim) for _ in range(depth)])

    def forward(self, x):
        x = self.patch_merging(x)
        x = self.blocks(x)
        return x
