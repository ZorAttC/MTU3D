from turtle import forward
import torch
import torch.nn as nn
import pdb, time
from torch_scatter import scatter_mean, scatter


class GeoAwarePooling(nn.Module):
    """Pool point features to super points.
    """
    def __init__(self, channel_proj=96):
        super().__init__()
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, channel_proj),
            nn.LayerNorm(channel_proj)
        )
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * channel_proj, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, 1, bias=False),
            nn.Sigmoid()
        )
    
    def scatter_norm(self, points, idx):
        ''' Normalize positions of same-segment in a unit sphere of diameter 1
        Code is copied from SPT
        '''
        min_segment = scatter(points, idx, dim=0, reduce='min')
        max_segment = scatter(points, idx, dim=0, reduce='max')
        diameter_segment = (max_segment - min_segment).max(dim=1).values
        center_segment = scatter(points, idx, dim=0, reduce='mean')
        center = center_segment[idx]
        diameter = diameter_segment[idx]
        points = (points - center) / (diameter.view(-1, 1) + 1e-2)
        return points, diameter_segment.view(-1, 1)

    def forward(self, pts_feat, sp_idx, all_xyz):
        """
        Args:
            pts_feat: (B,N, C) float Tensor, point features
            sp_idx: (B,N,) long Tensor, point to superpoint index
            all_xyz: (B, N, 3) float Tensor, point coordinates
        """
        import pudb; pudb.set_trace()
        B, N, C = pts_feat.shape
        all_xyz_flat = all_xyz.view(B * N, 3)
        pts_feat_flat = pts_feat.view(B * N, C)
        sp_idx_flat = sp_idx.view(B * N)
        
        # Adjust sp_idx to be global across batches
        sp_counts = []
        for i in range(B):
            sp_counts.append(sp_idx[i].max().item() + 1)
        offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(sp_counts[:-1]), 0)), device=sp_idx.device)
        sp_idx_flat = sp_idx.view(B * N) + offsets[torch.arange(B, device=sp_idx.device).repeat_interleave(N)]
        
        # Normalize positions
        all_xyz_norm, _ = self.scatter_norm(all_xyz_flat, sp_idx_flat)
        all_xyz_proj = self.pts_proj1(all_xyz_norm)
        all_xyz_segment = scatter(all_xyz_proj, sp_idx_flat, dim=0, reduce='max')
        all_xyz_concat = torch.cat([all_xyz_proj, all_xyz_segment[sp_idx_flat]], dim=-1)
        all_xyz_w = self.pts_proj2(all_xyz_concat) * 2
        
        sp_feat_flat = scatter_mean(pts_feat_flat * all_xyz_w, sp_idx_flat, dim=0) + all_xyz_segment
        
        # Split back to list of tensors per batch
        sp_cum = torch.cumsum(torch.tensor([0] + sp_counts), 0)
        sp_feat = [sp_feat_flat[sp_cum[i]:sp_cum[i+1]] for i in range(B)]
        all_xyz_w = all_xyz_w.view(B, N, 1)
        
        return sp_feat, all_xyz_w
