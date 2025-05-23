import torch.nn as nn

class PatchMerging(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 2x2 merging
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, H//2, W//2, 2, 2, C)
        x = x.view(B, H // 2, W // 2, 4 * C)  # concat 2x2 patches

        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C_out, H//2, W//2)
        return x
