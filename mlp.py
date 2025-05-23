import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

#    def forward(self, x):
#        x = F.gelu(self.fc1(x))
#        x = self.fc2(x)
#        return x
    
    def forward(self, x):
        # x: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # بازگشت به (B, C, H, W)
        return x