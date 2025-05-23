import torch.nn as nn
from swin_encoder import SwinEncoder
from swin_decoder import SwinDecoder

class TransUNet(nn.Module):
    def __init__(self, num_classes=1, patch_size=16):
        super().__init__()
        self.encoder = SwinEncoder(patch_size)
        self.decoder = SwinDecoder()
        self.seg_head = nn.Conv2d(1, num_classes, kernel_size=1)  # در صورت نیاز به چند کلاس

    def forward(self, x):
        x_encoded, skips = self.encoder(x)
        x_decoded = self.decoder(x_encoded, skips)
        x_out = self.seg_head(x_decoded)
        return x_out
