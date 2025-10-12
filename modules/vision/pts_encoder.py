from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_min

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling, MinkowskiMaxPooling

import modules.third_party.mask3d as mask3d_models
from modules.third_party.mask3d.common import conv
from modules.third_party.mask3d.helpers_3detr import GenericMLP
from modules.third_party.mask3d.position_embedding import PositionEmbeddingCoordsSine

from modules.build import VISION_REGISTRY
import time
from modules.layers.geo_aware_pooling import GeoAwarePooling

@VISION_REGISTRY.register()
class PCDESAMSegLevelEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, hidden_size, hlevels, freeze_backbone=False, dropout=0.1, geo_aware_pooling=True):
        super().__init__()
        # free backbone or not
        self.use_geo_aware_pooling = geo_aware_pooling
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.backbone = getattr(mask3d_models, "Res16UNet34C")(**backbone_kwargs)
        # 统计backbone参数量（以M为单位）
        num_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Backbone parameter count: {num_params / 1e6:.2f}M")
        self.scatter_fn = scatter_mean
        self.sizes = self.backbone.PLANES[-5:]
        self.hlevels = hlevels + [4] # 4 is for the last level, always used for mask seg features
        self.feat_proj_list = nn.ModuleList([
                                    nn.Sequential(
                                        nn.Linear(self.sizes[hlevel], hidden_size), 
                                        nn.LayerNorm(hidden_size),
                                        nn.Dropout(dropout)
                                    ) for hlevel in self.hlevels])
        self.pooltr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dilation=1, dimension=3)
        # Geo-aware pooling from ESAM
        if  self.use_geo_aware_pooling:
            self.geo_aware_pooling = GeoAwarePooling(channel_proj=96)

    def upsampling(self, feat, hlevel):
        n_pooltr = 4 - hlevel # 4 levels in totoal
        for _ in range(n_pooltr):
            feat = self.pooltr(feat)
        return feat
            
    def forward(self, x, voxel2segment, max_seg,data_dict=None):
        '''
        return:
            multi_scale_seg_feats: list of (B, max_seg, C), C is hidden_size
            pcds_features: ME.SparseTensor, (N, C2), C2
            pcds_w: (N, 1) geometry weight for each point
        '''
        # import pdb; pdb.set_trace()
        import pudb; pudb.set_trace()
        with self.context():
            # minkowski backbone
            
            pcds_features, aux = self.backbone(x)
            if self.use_geo_aware_pooling:
                voxel_feat = pcds_features.decomposed_features
                coord_list=[p[:,:3] for p in data_dict['raw_coordinates']]
                voxel_to_full_maps = data_dict['voxel_to_full_maps']
             
                pts2spidx = [v2s[v2p] for v2s, v2p in zip(data_dict['voxel2segment'], voxel_to_full_maps)]
                pts_feat_list=[v_feat[idx] for v_feat,idx in zip(voxel_feat,voxel_to_full_maps)]
                pts_feat_batched=torch.stack(pts_feat_list,dim=0).detach()
                pts_pos_batched=torch.from_numpy(np.stack(coord_list,axis=0)).cuda()
                pts2spidx_batched=torch.stack(pts2spidx,dim=0).detach()
                print("device of pts_pos_batched:",pts_pos_batched.device)
                _, pcds_w = self.geo_aware_pooling.forward(pts_feat_batched, pts2spidx_batched, pts_pos_batched)
            else:
                pcds_w = None
        multi_scale_seg_feats = []
        for hlevel, feat_proj in zip(self.hlevels, self.feat_proj_list):
            feat = aux[hlevel]
            feat = self.upsampling(feat, hlevel)
            assert feat.shape[0] == pcds_features.shape[0]
            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(feat.decomposed_features, voxel2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)
        return multi_scale_seg_feats , pcds_features ,pcds_w
