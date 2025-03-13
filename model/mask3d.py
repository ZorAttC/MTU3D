from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_min

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling

from model.build import MODEL_REGISTRY, BaseModel
from optim.utils import no_decay_param_group

import modules.third_party.mask3d as mask3d_models
from modules.third_party.mask3d.common import conv
from modules.third_party.mask3d.helpers_3detr import GenericMLP
from modules.third_party.mask3d.position_embedding import PositionEmbeddingCoordsSine
from modules.build import build_module, build_module_by_name
from functools import partial
from copy import deepcopy
from transformers import BertTokenizer, T5Tokenizer, AutoTokenizer
from data.datasets.constant import PromptType
from data.datasets.constant import CLASS_LABELS_200, CLASS_LABELS_REPLICA
from modules.utils import calc_pairwise_locs

@MODEL_REGISTRY.register()
class Mask3D(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
        self.use_gt_mask = cfg.model.get("use_gt_mask", False)
        
        # build voxel encoder
        for input in cfg.model.inputs:
            encoder = input + '_encoder'
            self.encoder = encoder
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        
        # build unified encoder
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        
        # build head
        self.seg_on_segments = cfg.model.seg_on_segments
        assert self.seg_on_segments == True # current we only support segmentations on segments
        for head in self.cfg.model.heads.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))
            
        # build query
        self.num_queries = cfg.model.num_queries
        hidden_size = cfg.model.voxel_encoder.args.query_dim
        self.query_projection = GenericMLP(
            input_dim=hidden_size,
            hidden_dims=[hidden_size],
            output_dim=hidden_size,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=hidden_size,
                                                       gauss_scale=1.0,
                                                       normalize=True)
         
    def forward(self, data_dict):
        # prepare
        voxel_features = data_dict['voxel_features'] # rgb + xyz
        voxel_coordinates = data_dict['voxel_coordinates']
        x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)

        # for Swin3D encoder
        if self.cfg.model.get(self.encoder, None).get('signal', False):
            if voxel_features.shape[1] > 3:
                if self.cfg.model.get(self.encoder, None).get('use_offset', False):
                    voxel_features[:, -3:] = voxel_coordinates[:, -3:] - voxel_coordinates[:, -3:].int()
            swin_sp = ME.SparseTensor(coordinates=voxel_coordinates.int(), features=voxel_features, device=voxel_features.device)
        else:
            swin_sp = ME.SparseTensor(coordinates=voxel_coordinates.int(), features=torch.ones_like(voxel_features).float(), device=voxel_features.device)
        colors = voxel_features[:, 0:3] / 1.001
        swin_coords_sp = ME.SparseTensor(
            features=torch.cat([voxel_coordinates, colors], dim=1), 
            coordinate_map_key=swin_sp.coordinate_map_key, 
            coordinate_manager=swin_sp.coordinate_manager
        )
        point2segment = data_dict['segment_to_voxel_maps']
        
        # backbone
        with self.optional_freeze():
            if 'Swin3D' in self.cfg.model.get(self.encoder, None).name:
                # mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates = self.voxel_encoder(swin_sp, swin_coords_sp, voxel_features, point2segment)
                mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates = self.voxel_encoder(swin_sp, swin_coords_sp, voxel_features, point2segment, voxel_coordinates[:, 0])
            else:
                mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates = self.voxel_encoder(x, voxel_features, point2segment)
        if self.use_gt_mask:
            mask_head_partial = partial(self.mask_head, mask_features=mask_features, mask_segments=mask_segments, ret_attn_mask=True, point2segment=point2segment, gt_attn_mask=data_dict['segment_masks'])
        else:
            mask_head_partial = partial(self.mask_head, mask_features=mask_features, mask_segments=mask_segments, ret_attn_mask=True, point2segment=point2segment)
        
        # build positional encoding and queries
        pos_encodings_pcd = self.get_multi_level_pos_encs(multi_scale_coordinates)
        sampled_coords = None
        if self.use_gt_mask:
            gt_coordinates = deepcopy(data_dict['obj_center'])
            query_masks = torch.ones((len(gt_coordinates), self.num_queries), dtype=torch.bool,device=gt_coordinates[0].get_device())
            for bid in range(len(gt_coordinates)):
                query_masks[bid, : gt_coordinates[bid].shape[0]] = False
            for bid in range(len(gt_coordinates)):
                assert gt_coordinates[bid].shape[0] < self.num_queries
                if gt_coordinates[bid].shape[0] < self.num_queries:
                    gt_coordinates[bid] = torch.cat([gt_coordinates[bid], torch.zeros(self.num_queries - gt_coordinates[bid].shape[0], 3).to(gt_coordinates[bid].get_device())], dim=0)
                else:
                    perm = torch.randperm(gt_coordinates[bid].shape[0])[:self.num_queries]
                    gt_coordinates[bid] = gt_coordinates[bid][perm]
            sampled_coords = torch.stack(gt_coordinates)
        else:
            gt_coordinates = deepcopy(data_dict['obj_center'])
            query_masks = torch.zeros((len(gt_coordinates), self.num_queries), dtype=torch.bool,device=gt_coordinates[0].get_device())
            fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(),
                                            self.num_queries).squeeze(0).long()
                    for i in range(len(x.decomposed_coordinates))]
            sampled_coords = torch.stack([coordinates.decomposed_features[i][fps_idx[i].long(), :]
                                        for i in range(len(fps_idx))])
        mins = torch.stack([coordinates.decomposed_features[i].min(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
        maxs = torch.stack([coordinates.decomposed_features[i].max(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
        query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs]) # Batch, Dim, queries
        query_pos = self.query_projection(query_pos).permute(0, 2, 1) 
        queries = torch.zeros_like(query_pos)
        
        # unifed encoder
        queries, predictions_class, predictions_mask = self.unified_encoder(queries, query_pos, multi_scale_features, pos_encodings_pcd, mask_head_partial, not self.training, query_masks)
        output_class, outputs_mask = self.mask_head(query_feat=queries, mask_features=mask_features, mask_segments=mask_segments, num_pooling_steps=0, ret_attn_mask=False, point2segment=point2segment)
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
    
        data_dict['output'] = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None,
            'queries': queries
        }
        return data_dict

    def get_opt_params(self):
        return [{'params': self.parameters(), 'lr': self.cfg.solver.lr}]

    def get_multi_level_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        
@MODEL_REGISTRY.register()
class Mask3DSegLevel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.memories = cfg.model.memories
        self.heads = cfg.model.heads
        self.num_queries = cfg.model.num_queries
        self.hidden_size = cfg.model.hidden_size
        self.use_gt_mask = cfg.model.get("use_gt_mask", False)
        self.use_gt_mask_eval = cfg.eval.get("use_gt_mask", False)
        self.use_offline_voxel_fts = cfg.model.get("use_offline_voxel_fts", False)

        self.inputs = self.memories[:]
        
        self.prompt_types = ['txt']
        for input in self.inputs:
            if input == 'prompt':
                for prompt_type in self.prompt_types: # only text prompt for now
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
            else:
                encoder = input + '_encoder'
                setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
            
        hidden_size = self.hidden_size
        self.coord_encoder = CoordinateEncoder(hidden_size)
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type if hasattr(self.cfg.model, 'obj_loc') else 'center' 
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim if hasattr(self.cfg.model, 'obj_loc') else 5

        # whether use for loop for test
        self.test_prompt_scannet200 = cfg.eval.get("test_prompt_scannet200", False)
        self.test_prompt_replica = cfg.eval.get("test_prompt_replica", False)
        self.filter_out_classes = cfg.eval.get("filter_out_classes", [0, 2])
        if self.test_prompt_replica or self.test_prompt_scannet200:
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def prompt_encoder(self, data_dict):
        prompt = data_dict['prompt']
        prompt_mask = data_dict['prompt_masks']
        prompt_type = data_dict['prompt_type']
        prompt_feat = torch.zeros(prompt.shape + (self.hidden_size,), device=prompt.device)
        for type in self.prompt_types:
            encoder = getattr(self, type + '_encoder')
            idx = prompt_type == getattr(PromptType, type.upper())
            input = prompt[idx]
            mask = prompt_mask[idx]
            if type == 'txt':
                feat = encoder(input.long(), mask)
            else:
                raise NotImplementedError
            prompt_feat[idx] = feat
        return prompt_feat, prompt_mask.logical_not()

    def forward(self, data_dict):
        # prepare
        voxel_features = data_dict['voxel_features']
        voxel_coordinates = data_dict['voxel_coordinates']
        x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
        voxel2segment = data_dict['voxel2segment']
        coordinates = data_dict['coordinates']
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        seg_center = data_dict['seg_center']
        use_gt_mask = (self.training and self.use_gt_mask) or (not self.training and self.use_gt_mask_eval)
        data_dict['use_gt_mask'] = use_gt_mask
        
        seg_pad_masks = data_dict['seg_pad_mask'].logical_not()
        seg_pos = self.coord_encoder(seg_center, input_range=[coord_min, coord_max])
        mv_seg_pad_masks = torch.logical_or(seg_pad_masks, data_dict['mv_seg_pad_mask'].logical_not())
        pc_seg_pad_masks = torch.logical_or(seg_pad_masks, data_dict['pc_seg_pad_mask'].logical_not())

        input_dict = {}
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'prompt':
                feat, mask = self.prompt_encoder(data_dict)
            elif input == 'mv':
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = mv_seg_pad_masks
                pos = seg_pos
            elif input == 'pc':
                feat = self.pc_encoder(obj_feats = data_dict['pc_seg_fts'])
                mask = pc_seg_pad_masks
                pos = seg_pos
            elif input == 'voxel':
                if self.use_offline_voxel_fts:
                    feat = data_dict['voxel_seg_fts']
                else:
                    feat = self.voxel_encoder(x, voxel2segment, max_seg=seg_center.shape[1])
                voxel_seg_feature = feat.copy()
                mask = seg_pad_masks
                pos = seg_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            input_dict[input] = [feat, mask, pos]
            
        if not self.test_prompt_scannet200 or self.training:
            # build query
            if use_gt_mask:
                query_masks = data_dict['obj_pad_mask'].logical_not()
                sampled_coords = data_dict['obj_center']
            else:
                query_masks = None
                voxel_coordinates = x.decomposed_coordinates
                fps_idx = [furthest_point_sample(voxel_coordinates[i][None, ...].float(), self.num_queries).squeeze(0).long()
                        for i in range(len(voxel_coordinates))]
                sampled_coords = torch.stack([coordinates[i][fps_idx[i]] for i in range(len(fps_idx))])
            query_pos = self.coord_encoder(sampled_coords, input_range=[coord_min, coord_max])
            query = torch.zeros_like(query_pos)
            input_dict['query'] = [query, query_masks, query_pos]
            
            if self.unified_encoder.spatial_selfattn:
                pairwise_locs = calc_pairwise_locs(sampled_coords, None, pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True, spatial_dim=self.spatial_dim)
            else:
                pairwise_locs = None
                
            # feature for segment matching
            seg_fts_for_match = []
            for input in self.inputs:
                if input in ['voxel', 'mv', 'pc']:
                    feats = input_dict[input][:]
                    if isinstance(feats[0], list):
                        assert input == 'voxel'
                        feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                    seg_fts_for_match.append(feats)
            
            # build mask head
            gt_attn_mask = data_dict['gt_attn_mask'] if use_gt_mask else None
            mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, seg_pad_masks=seg_pad_masks, 
                                        gt_attn_mask=gt_attn_mask, query_pos=query_pos)

            # unifed encoder
            query, predictions_class, predictions_mask = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)

            # grounding
            if hasattr(self, 'ground_head'):
                ground_logits = self.ground_head(query, query_masks)
                data_dict['ground_logits'] = ground_logits
        else:
            # get total number
            if self.test_prompt_scannet200:
                class_label = [f"a {class_name} in a scene" for class_name in CLASS_LABELS_200]
            elif self.test_prompt_replica:
                class_label = [f"a {class_name} in a scene" for class_name in CLASS_LABELS_REPLICA]
            total_number = len(class_label) + 1
            
            # build prompt
            prompt = []
            for class_name in class_label:
                prompt.append(class_name)
            prompt.append("")
            encoded_input = self.tokenizer(prompt, padding='max_length', return_tensors="pt", truncation=True, max_length=50)
            prompt_feats = self.txt_encoder(encoded_input.input_ids.to(voxel_features.device), encoded_input.attention_mask.bool().to(voxel_features.device))
            prompt_mask = encoded_input.attention_mask.bool().logical_not().to(voxel_features.device) 
            input_dict['prompt'] = [prompt_feats, prompt_mask, None]
            
            # repeat
            if use_gt_mask:
                query_masks = self.expand_tensor(data_dict['obj_pad_mask'].logical_not(), total_number)
                sampled_coords = data_dict['obj_center']
            else:
                query_masks = None
                voxel_coordinates = x.decomposed_coordinates
                fps_idx = [furthest_point_sample(voxel_coordinates[i][None, ...].float(), self.num_queries).squeeze(0).long()
                        for i in range(len(voxel_coordinates))]
                sampled_coords = torch.stack([coordinates[i][fps_idx[i]] for i in range(len(fps_idx))])
            query_pos = self.expand_tensor(self.coord_encoder(sampled_coords, input_range=[coord_min, coord_max]), total_number)
            query = self.expand_tensor(torch.zeros_like(query_pos), total_number)
            input_dict['query'] = [query, query_masks, query_pos]
            
            if self.unified_encoder.spatial_selfattn:
                pairwise_locs = self.expand_tensor(calc_pairwise_locs(sampled_coords, None, pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True, spatial_dim=self.spatial_dim), total_number)
            else:
                pairwise_locs = None
                
            # feature for segment matching
            seg_fts_for_match = []
            for input in self.inputs:
                if input in ['voxel', 'mv', 'pc']:
                    feats = input_dict[input][:]
                    if isinstance(feats[0], list):
                        assert input == 'voxel'
                        feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                    feats[0] = self.expand_tensor(feats[0], total_number)
                    seg_fts_for_match.append(feats)
            
            # build mask head
            gt_attn_mask = data_dict['gt_attn_mask'].repeat_interleave(total_number, 0) if use_gt_mask else None
            mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, seg_pad_masks=seg_pad_masks.repeat_interleave(total_number, 0), 
                                        gt_attn_mask=gt_attn_mask, query_pos=query_pos)

            # repeat
            for input in self.inputs:
               for idx in range(3):
                   if input == 'voxel' and idx == 0:
                       for l in range(len(input_dict[input][idx])):
                           input_dict[input][idx][l] = self.expand_tensor(input_dict[input][idx][l], total_number)
                   else:
                       input_dict[input][idx] = self.expand_tensor(input_dict[input][idx], total_number)

            # unifed encoder
            query, predictions_class, predictions_mask = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)

            # grounding
            if hasattr(self, 'ground_head'):
                ground_logits = self.ground_head(query, query_masks)
                data_dict['ground_logits'] = ground_logits[total_number -1][None, ...]
                
            # reorganize
            prompt_logits = ground_logits.permute(1, 0)[None, ..., :total_number - 1]
            for filter_out_id in self.filter_out_classes:
                prompt_logits[..., filter_out_id] = float("-inf")
            prompt_masks = [predictions_mask[-1].permute(1, 2, 0)[..., :total_number-1]]
                
                
        data_dict['output'] = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'prompt_logits': prompt_logits if (self.test_prompt_scannet200 or self.test_prompt_replica) and not self.training else None,
            'prompt_masks': prompt_masks if (self.test_prompt_scannet200 or self.test_prompt_replica) and not self.training else None,
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'voxel_seg_feature': voxel_seg_feature, 
            'voxel_seg_mask': seg_pad_masks
            # 'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None,
            # 'queries': query,
            # 'voxel_mask_segments': voxel_mask_segments,
            # 'voxel_multi_scale_features': voxel_multi_scale_features,
            # 'voxel_multi_scale_coordinates': voxel_multi_scale_coordinates,
            # 'voxel_multi_scale_segment_masks': voxel_multi_scale_segment_masks
        }
        return data_dict

    def get_opt_params(self):
        return [{'params': self.parameters(), 'lr': self.cfg.solver.lr}]
    
    def expand_tensor(self, x, num):
        return x.expand([num] + [-1] * (len(x.shape) - 1)) if x is not None else None

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        

class CoordinateEncoder(nn.Module):
    def __init__(self, hidden_size, use_projection=True):
        super().__init__()
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=hidden_size, gauss_scale=1.0, normalize=True)
        if use_projection:
            self.feat_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
    
    def forward(self, coords, input_range):
        with autocast(enabled=False):
            pos = self.pos_enc(coords, input_range=input_range).permute(0, 2, 1)
        if hasattr(self, 'feat_proj'):
            pos = self.feat_proj(pos)
        return pos