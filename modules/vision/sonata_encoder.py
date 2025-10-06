import os
import json
import einops
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import get_mlp_head, get_mixup_function
from modules.weights import _init_weights_bert
from modules.build import VISION_REGISTRY
import modules.third_party.sonata as sonata
from torch_scatter import scatter_max, scatter_mean, scatter_min
try:
    import flash_attn
except ImportError:
    flash_attn = None
import copy

@VISION_REGISTRY.register()
class Sonata3DSegLevelEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, hidden_size, hlevels, freeze_backbone=False, dropout=0.1):
        super().__init__()
        # free backbone or not
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.backbone = sonata.load("sonata", repo_id="facebook/sonata").cuda()
        self.scatter_fn = scatter_mean
        self.sizes = [512,896,1088,1184,1232]
        self.hlevels = hlevels + [4] # 4 is for the last level, always used for mask seg features
        self.feat_proj_list = nn.ModuleList([
                                    nn.Sequential(
                                        nn.Linear(size, hidden_size), 
                                        nn.LayerNorm(hidden_size),
                                        nn.Dropout(dropout)
                                    ) for size in self.sizes])
\
    def forward(self, x, point2segment, max_seg):
        multi_scale_feats_from_fine_to_coarse = []
        with self.context():
            point = self.backbone(x)
            # From fine to coarse, collect all feature maps and their inverse indices
            while "pooling_parent" in point.keys():
                # Store a copy of the current (finer) point cloud and the inverse map to its parent
                inverse = point.pop("pooling_inverse")
                multi_scale_feats_from_fine_to_coarse.append({'map': point, 'inverse': inverse})
                
                parent = point.pop("pooling_parent")
                # The original implementation concatenates features during downsampling.
                # We will do this on the fly during upsampling instead.
                point = parent
            # Add the last, coarsest feature map
            multi_scale_feats_from_fine_to_coarse.append({'map': point, 'inverse': None})

        multi_scale_feats_from_coarse_to_fine = list(reversed(multi_scale_feats_from_fine_to_coarse))
        
        multi_scale_seg_feats = []

        # Iterate from coarse to fine
        for i in range(len(multi_scale_feats_from_coarse_to_fine)):
            feat_map = multi_scale_feats_from_coarse_to_fine[i]['map']
            feat_proj = self.feat_proj_list[i]

            # Decompose features by batch manually
            num_batches = feat_map.batch.max().item() + 1
            decomposed_features = []
            for batch_idx in range(num_batches):
                batch_mask = (feat_map.batch == batch_idx)
                decomposed_features.append(feat_map.feat[batch_mask])
            
            import pdb; pdb.set_trace()
            # Calculate segment features for the current level
            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)

            # Upsample and concatenate features for the next (finer) level
            if i + 1 < len(multi_scale_feats_from_coarse_to_fine):
                next_level_info = multi_scale_feats_from_coarse_to_fine[i+1]
                inverse_map = next_level_info['inverse']
                next_feat_map = next_level_info['map']
                
                # Use inverse_map to bring current features to the finer resolution of the next level
                upsampled_feat = feat_map.feat[inverse_map]
                
                # Concatenate with the original features of the next level
                next_feat_map.feat = torch.cat([next_feat_map.feat, upsampled_feat], dim=-1)

        return multi_scale_seg_feats


@VISION_REGISTRY.register()
class Sonata3DEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, hidden_size, hlevels, freeze_backbone=False, dropout=0.1):
        super().__init__()
        # free backbone or not
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.backbone = sonata.load("sonata", repo_id="facebook/sonata").cuda()
        self.scatter_fn = scatter_mean
        self.sizes = [512,896,1088,1184,1232]
        self.feat_proj = nn.Sequential(
            nn.Linear(1232, hidden_size), 
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, point2segment, max_seg):

        with self.context():
            point = self.backbone(x)
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                # Deep copy point to avoid modification affecting the original dict
              
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent

        # Reshape point.feat based on batch
        num_batches = point.batch.max().item() + 1
        feat_list = []
        for i in range(num_batches):
            batch_mask = (point.batch == i)
            feat_list.append(point.feat[batch_mask])
        
        import pdb; pdb.set_trace()
        multi_scale_seg_feats = []
        
        # 这里假设sonata最后一层的索引和point2segment是一一对应的
        batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(feat_list, point2segment)]
        batch_feat = torch.stack(batch_feat)
        batch_feat = self.feat_proj(batch_feat)
        multi_scale_seg_feats.append(batch_feat)
        
        return multi_scale_seg_feats