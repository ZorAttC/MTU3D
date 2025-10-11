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

        sonata_version = backbone_kwargs.get("sonata_version", "sonata")
        custom_config = dict(
            mask_token=False,
            freeze_encoder=backbone_kwargs.get("freeze_encoder", False),
            dec_depths= backbone_kwargs.get("dec_depths", [1,1,1,1]),
            dec_channels= backbone_kwargs.get("dec_channels", [64, 64, 128, 256]),
            dec_num_head= backbone_kwargs.get("dec_num_head", [4, 4, 8, 16]),
            enc_mode= False
        )
        self.backbone = sonata.load(sonata_version, repo_id="facebook/sonata",custom_config=custom_config).cuda()
        
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.scatter_fn = scatter_mean
        self.sizes = custom_config['dec_channels']
        self.hlevels = [i for i in range(len(self.sizes))]
        self.feat_proj_list = nn.ModuleList([
                                    nn.Sequential(
                                        nn.Linear(size, hidden_size), 
                                        nn.LayerNorm(hidden_size),
                                        nn.Dropout(dropout)
                                    ) for size in self.sizes])

    def forward(self, x, point2segment, max_seg):
        with self.context():
            downsample_point = self.backbone(x)
        
        scaled_multi_layer_feat = []
        inverse_map = []
        scaled_multi_layer_feat.append(downsample_point.feat)
        batch = downsample_point.batch
            
        
        while "unpooling_parent" in downsample_point.keys():
            parent = downsample_point.pop("unpooling_parent")
            inverse = parent.pop("pooling_inverse")
            downsample_point = parent
            
            
            inverse_map.append(inverse)
            # Apply all inverse mappings in reverse order to upsample the features
            upsampled_feat = downsample_point.feat
            for i in range(len(inverse_map)):
                upsampled_feat = upsampled_feat[inverse_map[len(inverse_map)-1 - i]]
            scaled_multi_layer_feat.append(upsampled_feat)

            
        multi_scale_seg_feats = []
        # Iterate through each level `i` to generate a feature vector for it
        for i in range(len(self.sizes)):
            
            feat = scaled_multi_layer_feat[i]
            # Now decompose the upsampled features by batch at the finest level
            num_batches = batch.max().item() + 1
            decomposed_features = []
            for batch_idx in range(num_batches):
                batch_mask = (batch == batch_idx)
                decomposed_features.append(feat[batch_mask])

            # Use the projection layer corresponding to the current level `i`
            feat_proj = self.feat_proj_list[i]

            # Calculate segment features using the original point2segment mapping, which corresponds to the finest level
            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)
        multi_scale_seg_feats.reverse()

        return multi_scale_seg_feats    
@VISION_REGISTRY.register()
class Sonata3DSegLevelEncoder_no_decoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, hidden_size, hlevels, freeze_backbone=False, dropout=0.1):
        super().__init__()

        sonata_version = backbone_kwargs.get("sonata_version", "sonata")
        custom_config = dict(
            mask_token=False,
            freeze_encoder=freeze_backbone,
            dec_depths= [1,1,1,1],
            dec_channels= [64, 64, 128, 256],
            dec_num_head= [4, 4, 8, 16],
            enc_mode= False
        )
        self.backbone = sonata.load(sonata_version, repo_id="facebook/sonata",custom_config=custom_config).cuda()
        
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.feat_concat_upsample = True
        self.scatter_fn = scatter_mean
        self.sizes = [48, 96, 192, 384, 512] if sonata_version=="sonata" else [32,64,128,256,512]
        self.hlevels = hlevels + [4] # 4 is for the last level, always used for mask seg features
        self.feat_proj_list = nn.ModuleList([
                                    nn.Sequential(
                                        nn.Linear(size, hidden_size), 
                                        nn.LayerNorm(hidden_size),
                                        nn.Dropout(dropout)
                                    ) for size in reversed(self.sizes)])

    def forward(self, x, point2segment, max_seg):
        # The backbone returns a list of feature maps.
        # Per user: Index 0 has the fewest points (coarsest), and the last index has the most points (finest).
        multi_scale_feats = []
        with self.context():
            point = self.backbone(x)
            # Collect all feature maps and their inverse indices.
            # The `inverse` map at index `j` is used to upsample from level `j-1` to `j`.
            while "pooling_parent" in point.keys():
                inverse = point.pop("pooling_inverse")
                multi_scale_feats.append({'map': point, 'inverse': inverse})
                parent = point.pop("pooling_parent")
                point = parent
            # Add the last feature map (coarsest, which has no parent)
            multi_scale_feats.append({'map': point, 'inverse': None})
        
      
      
        multi_scale_seg_feats = []
        # multi_scale_feats[0]: 点数最少 512通道   multi_scale_feats[4]: 点数最多 48通道
        # Iterate through each level `i` to generate a feature vector for it (0=coarsest, 4=finest)
        for i in range(len(multi_scale_feats)):
            
            level_maps = [info['map'] for info in multi_scale_feats]

            # aggregated_feat = level_maps[i].feat
            # for j in range(i,len(level_maps)-1):
                
            #     inverse_map = multi_scale_feats[j]['inverse']
            #     map_feat = aggregated_feat
            #     upsampled_feat = map_feat[inverse_map]
            #     # Concatenate with the current level's original features
            #     aggregated_feat = torch.cat([level_maps[j+1].feat, upsampled_feat], dim=-1)
            upsampled_feat = level_maps[i].feat
            for j in range(i,len(level_maps)-1):
                
                inverse_map = multi_scale_feats[j]['inverse']
                upsampled_feat = upsampled_feat[inverse_map]
                
            
            
            # Now decompose the upsampled features by batch at the finest level
            finest_map = level_maps[len(level_maps)-1]
            num_batches = finest_map.batch.max().item() + 1
            upsampled_decomposed_features = []
            for batch_idx in range(num_batches):
                batch_mask = (finest_map.batch == batch_idx)
                upsampled_decomposed_features.append(upsampled_feat[batch_mask])

            # Use the projection layer corresponding to the current level `i`
            feat_proj = self.feat_proj_list[i]

            # Calculate segment features using the original point2segment mapping, which corresponds to the finest level
            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(upsampled_decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)

      
        output_feats = [multi_scale_seg_feats[i] for i in self.hlevels]
        
        output_feats.reverse()

        return output_feats


@VISION_REGISTRY.register()
class Sonata3DEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, hidden_size, hlevels, freeze_backbone=False, dropout=0.1):
        super().__init__()
        # free backbone or not
        self.context = torch.no_grad if freeze_backbone else nullcontext
        custom_config = dict(
            freeze_encoder=freeze_backbone
        )
        self.backbone = sonata.load("sonata", repo_id="facebook/sonata",custom_config=custom_config).cuda()
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