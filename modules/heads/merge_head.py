import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from modules.build import HEADS_REGISTRY

@HEADS_REGISTRY.register()
class MergeHead(nn.Module):
    def __init__(self, in_channels, out_channels, d_hidden=512):
        super().__init__()   
        self.net = nn.Sequential(
            nn.Linear(in_channels, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, out_channels),
            nn.LayerNorm(out_channels),
        )
    
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            res = F.normalize(self.net(x), p=2, dim=-1)
            return res
        results = []
        for data in x:
            res = F.normalize(self.net(data), p=2, dim=-1)
            results.append(res)
        return results