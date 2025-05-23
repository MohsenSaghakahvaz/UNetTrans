import torch.nn as nn

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        """
        x: (num_windows*B, N, C), N=window_size*window_size
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        return self.proj(out)

def window_partition(x, window_size):
    """
    x: (B, C, H, W)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_w_h, num_w_w, win_h, win_w, C)
    windows = x.view(-1, window_size, window_size, C)  # (num_windows*B, win_h, win_w, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, num_w_h, win_h, num_w_w, win_w)
    x = x.view(B, -1, H, W)
    return x
