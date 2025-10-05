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
import spconv.pytorch as spconv

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
        # Spconv's inverse convolution for upsampling
        self.upsamplers = nn.ModuleList([
            spconv.SparseInverseConv3d(self.sizes[i], self.sizes[i], kernel_size=2, indice_key=f"spconv_down_{i-1}") for i in range(1, 5)
        ])

    def upsampling(self, feat, target_level):
        # feat is a SparseConvTensor from a lower level
        # target_level is the destination level (e.g., 4 for the highest resolution)
        
        # Infer current level from indice_key
        indice_key = feat.indice_key
        if indice_key.startswith('subm'): # Base level
            current_level = 0
        else: # e.g., spconv_down_0
            current_level = int(indice_key.split('_')[-1]) + 1
        
        num_upsamples = target_level - current_level
        
        # Apply upsampling
        for i in range(num_upsamples):
            # The indice_key for SparseInverseConv3d should correspond to the downsampling that created the features
            # E.g., to reverse "spconv_down_2", we need to upsample from level 3 to 4, using the key "spconv_down_2"
            upsample_level = current_level + i
            if upsample_level < 4: # Max 4 levels of downsampling, so 4 upsamplers
                 feat = self.upsamplers[upsample_level](feat)
        return feat
            
    def forward(self, x, point2segment, max_seg):
        with self.context():
            # The new backbone returns a list of features from different levels of the decoder
            # The list is ordered from highest resolution (most points) to lowest resolution (fewest points)
            multi_scale_feats = self.backbone(x)

        multi_scale_seg_feats = []
        
        # The first feature map in the list has the highest resolution and is our target
        target_feat = multi_scale_feats[0]
        target_indices = target_feat.indices
        target_num_points = target_indices.shape[0]

        for i, (feat, feat_proj) in enumerate(zip(multi_scale_feats, self.feat_proj_list)):
            # All features need to be upsampled to the highest resolution (level 4)
            # The upsampling logic uses SparseInverseConv3d which should align the points correctly.
            upsampled_feat = self.upsampling(feat, 4)
            
            assert upsampled_feat.features.shape[0] == target_num_points, f"Point count mismatch after upsampling for feature {i}. Expected {target_num_points}, got {upsampled_feat.features.shape[0]}"

            # Decompose features by batch. After upsampling, the indices should match the target feature's indices.
            decomposed_feat_tensor = spconv.SparseConvTensor(
                features=upsampled_feat.features,
                indices=target_indices,
                spatial_shape=target_feat.spatial_shape,
                batch_size=x.batch_size
            )
            
            decomposed_features = decomposed_feat_tensor.decomposed_features

            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)
        
        # Reverse the order as requested, so it's from coarse to fine
        multi_scale_seg_feats.reverse()
        
        return multi_scale_seg_feats

    def upsampling(self, feat, target_feat_shape, target_indices):
        # Upsample feat to match the point cloud with the most points
        num_upsamples = target_feat_shape[0] - feat.spatial_shape[0]
        for _ in range(num_upsamples):
            feat = self.pooltr(feat)
        
        # Reorder points to match the original order of the target feature
        # Create a zero tensor with the target shape
        upsampled_features = torch.zeros((target_indices.shape[0], feat.features.shape[1]), 
                                         device=feat.features.device, 
                                         dtype=feat.features.dtype)
        # Create a map from original indices to a continuous range
        # This is needed because the indices from spconv are not guaranteed to be continuous
        # and we need to place the features in the correct positions in the upsampled_features tensor.
        # We assume target_indices are sorted, so we can do this mapping.
        # A more robust way might be needed if indices are not sorted.
        
        # A simple and potentially slow way to map indices
        # This assumes batch_idx is the first column in indices
        
        # Let's assume the batch size is 1 for simplicity or that indices are globally unique for now.
        # A more complex logic is needed for batch > 1 if indices are not globally unique across batches.
        
        # Assuming indices are [batch_idx, z, y, x] and we care about point order
        # The indices tensor from spconv contains the coordinates of the points.
        # We need to match the points from `feat` to `target_indices`.
        # A simple reordering might not work if the point sets are different.
        
        # The user's description implies that the lower-res features are subsets of higher-res ones.
        # `pooling_inverse` in the user's example suggests we can map back.
        # Since the new backbone doesn't provide `pooling_inverse`, we have to rely on coordinates.
        
        # Let's try a simpler approach first. After upsampling, the points should align.
        # If the upsampling is correct, the indices of the upsampled `feat` should be a subset of `target_indices`.
        
        # After upsampling, feat.indices should have the same spatial size as target_feat_shape
        # but the number of points might be different.
        # The goal is to have a dense feature tensor corresponding to target_indices
        
        # Let's rethink. The upsampling should bring it to the same spatial resolution.
        # The number of active sites (points) in `feat` will be smaller than in `target_feat`.
        # We need to create a feature tensor that has features for all points in `target_indices`.
        # For points in `target_indices` that are also in `feat.indices`, we use `feat.features`.
        # For other points, the features will be zero.
        
        # This is getting complicated. Let's look at the original code again.
        # `feat = self.upsampling(feat, hlevel)`
        # `assert feat.shape[0] == pcds_features.shape[0]`
        # This implies the number of points should be the same.
        
        # The new `upsampling` should do the same.
        # After `self.pooltr`, the spatial shape increases, but the number of points might not match.
        
        # Let's go with a hash-map based reordering. It's safer.
        # Convert target_indices to a hashable format to create a lookup map.
        target_indices_map = {tuple(coord.tolist()): i for i, coord in enumerate(target_indices)}
        
        # Create the output tensor
        output_features = torch.zeros((len(target_indices), feat.features.shape[1]), 
                                      device=feat.features.device, dtype=feat.features.dtype)
                                      
        # Place features from `feat` into the correct positions in `output_features`
        for i, coord in enumerate(feat.indices):
            if tuple(coord.tolist()) in target_indices_map:
                target_idx = target_indices_map[tuple(coord.tolist())]
                output_features[target_idx] = feat.features[i]

        return output_features

    def forward(self, x, point2segment, max_seg):
        with self.context():
            # spconv backbone
            multi_scale_feats = self.backbone(x)

        multi_scale_seg_feats = []
        
        # The first feature map has the most points, use it as target
        target_feat = multi_scale_feats[0]
        target_feat_shape = target_feat.spatial_shape
        target_indices = target_feat.indices

        for feat, feat_proj in zip(multi_scale_feats, self.feat_proj_list):
            # upsample to the same point cloud resolution
            upsampled_feat_features = self.upsampling(feat, target_feat_shape, target_indices)
            assert upsampled_feat_features.shape[0] == target_indices.shape[0]
            
            # Decompose features by batch
            # We need to manually split the features based on batch indices in target_indices
            decomposed_features = []
            for b in range(x.batch_size):
                batch_mask = target_indices[:, 0] == b
                decomposed_features.append(upsampled_feat_features[batch_mask])

            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)
        
        multi_scale_seg_feats.reverse()
        return multi_scale_seg_feats
            
    def forward(self, x, point2segment, max_seg):
        with self.context():
            # spconv backbone
            multi_scale_feats = self.backbone(x)

        multi_scale_seg_feats = []
        
        # The first feature map has the most points, use it as target
        target_feat = multi_scale_feats[0]
        target_shape = target_feat.spatial_shape
        target_indices = target_feat.indices

        for feat, feat_proj in zip(multi_scale_feats, self.feat_proj_list):
            upsampled_feat = self.upsampling(feat, target_shape, target_indices)
            assert upsampled_feat.shape[0] == target_indices.shape[0]
            
            # Create a SparseConvTensor from the upsampled features to decompose them by batch
            decomposed_feat_tensor = spconv.SparseConvTensor(upsampled_feat, target_indices, target_shape, x.batch_size)
            
            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(decomposed_feat_tensor.decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)
        
        multi_scale_seg_feats.reverse()
        return multi_scale_seg_feats


