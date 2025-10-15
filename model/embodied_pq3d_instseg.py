from copy import copy
from functools import partial
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from data.datasets.constant import PromptType

from modules.build import build_module_by_name
from modules.utils import calc_pairwise_locs
from model.build import MODEL_REGISTRY, BaseModel
from optim.utils import no_decay_param_group
from model.mask3d import CoordinateEncoder
from torch_scatter import scatter_mean, scatter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
def scatter_norm(points, idx):
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

@MODEL_REGISTRY.register()
class EmbodiedPQ3DInstSegModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories
        self.heads = cfg.model.heads
        self.inputs = self.memories[:]
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        self.init_query_by_feat = cfg.model.get('init_query_by_feat', False)
        self.add_geometry_to_segment = cfg.model.get('add_geometry_to_segment', False)
        
        # build feature encoder
        for input in self.inputs:
            encoder = input + '_encoder'
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        # add geometry to segment
        if self.add_geometry_to_segment:
            self.pts_proj1 = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.feat_proj = nn.Sequential(
                nn.Linear(hidden_size + 3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        if self.init_query_by_feat:
            self.input_weights = nn.ParameterDict({
                input: nn.Parameter(torch.ones(1)) for input in self.inputs
            })
         
    def forward(self, data_dict):
        input_dict = {}
        # build query
        mask = data_dict['query_pad_masks'].logical_not()
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        feat = torch.zeros_like(query_pos)
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center']
        fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'mv':
                # import pdb; pdb.set_trace()
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'voxel':
                # import pudb; pudb.set_trace() 
                voxel_features = data_dict['voxel_features']
                voxel_coordinates = data_dict['voxel_coordinates']
                x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
                voxel2segment = data_dict['voxel2segment']
                feat = self.voxel_encoder(x, voxel2segment, max_seg=fts_locs.shape[1])
                if self.add_geometry_to_segment:
                    for bid in range(len(voxel2segment)):
                        sp_idx = voxel2segment[bid]
                        all_xyz = data_dict['coordinates'][bid]
                        norm_xyz, _ = scatter_norm(all_xyz, sp_idx)
                        all_xyz_segment = scatter(self.pts_proj1(norm_xyz), sp_idx, dim=0, reduce='max', dim_size=fts_locs.shape[1])
                        for i in range(len(feat)):
                            feat[i][bid] = feat[i][bid] + all_xyz_segment
                    for i in range(len(feat)):        
                        feat[i] = self.feat_proj(torch.cat([feat[i], fts_locs], dim=-1)) 
                mask = data_dict['seg_pad_masks'].logical_not()
                data_dict['voxel_feat'] = {'feat': feat[-1].detach().cpu(), 'mask': mask.detach().cpu()}
                pos = fts_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            input_dict[input] = [feat, mask, pos]
        # build offline attention mask for guided mask training
        offline_attn_masks = None
        # generate features for mask head
        seg_fts_for_match = []
        for input in self.inputs:
            if input in ['voxel', 'mv']:
                feats = copy(input_dict[input][:])
                if isinstance(feats[0], list):
                    assert input == 'voxel'
                    feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                seg_fts_for_match.append(feats)                 
        # import pdb; pdb.set_trace()
        # build mask head
        if hasattr(self, 'mask_head'):
            mask_head_partial = partial(self.mask_head, query_locs=query_locs, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                        offline_attn_masks=offline_attn_masks, skip_prediction=self.skip_query_encoder_mask_pred)
        else:
            mask_head_partial = None
        # generate features for spatial attention
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
        # init query
        if self.init_query_by_feat:
            input_query_feat_list = []
            for input in self.inputs:
                if input == 'voxel':
                    segment_feature = input_dict['voxel'][0][-1]
                else:
                    segment_feature = input_dict[input][0]
                query_selection_ids = data_dict['query_selection_ids']
                query_feat_list = []
                for bid in range(len(query_selection_ids)):
                    query_feat = segment_feature[bid][query_selection_ids[bid]]
                    query_feat_list.append(query_feat)
                query_feat = pad_sequence(query_feat_list, batch_first=True) * self.input_weights[input]
                input_query_feat_list.append(query_feat)
            query = torch.stack(input_query_feat_list, dim=-1).sum(dim=-1)
            assert query.shape == input_dict['query'][0].shape, f"Query shape {query.shape} does not match input shape {input_dict['query'][0].shape}"
            input_dict['query'] = (query, input_dict['query'][1], input_dict['query'][2])
        # unified encoding    
        # import pudb; pudb.set_trace()                       
        query, predictions_score, predictions_class, predictions_mask, predictions_box = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)
        data_dict['query_feat'] = query
        
        # task head
        for head in self.heads:
            if head == 'mask':
                if self.skip_query_encoder_mask_pred:
                    mask_head_partial = partial(self.mask_head, query_locs=query_locs, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                    offline_attn_masks=offline_attn_masks, skip_prediction=False)
                    predictions_score = []
                    predictions_class = []
                    predictions_mask = []
                    predictions_box = []
                pred_scores, pred_logits, pred_masks, pred_boxes, _ = mask_head_partial(query=query)
                predictions_score.append(pred_scores)
                predictions_class.append(pred_logits)
                predictions_mask.append(pred_masks)
                predictions_box.append(pred_boxes)
                data_dict['predictions_score'] = predictions_score
                data_dict['predictions_class'] = predictions_class
                data_dict['predictions_mask'] = predictions_mask
                data_dict['predictions_box'] = predictions_box  
                continue
            elif head == 'openvocab':
                openvocab_query_feat = getattr(self, 'openvocab_head')(query)
                data_dict['openvocab_query_feat'] = openvocab_query_feat
            else:
                raise NotImplementedError(f"Unknow head type: {head}")
       
        return data_dict
    def params_statistics(self):
        """
        统计模型中每个主要部分的参数数量。
        
        Args:
            model: PyTorch模型实例
        """
        print("Parameter Statistics:")
        total_params = 0
        for name, module in self._modules.items():
            module_params = sum(p.numel() for p in module.parameters())
            total_params += module_params
            print(f"  {name}: {module_params:,} parameters")
        print(f"  Total: {total_params:,} parameters")

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            
            # 只收集需要梯度的参数
            trainable_params = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                optimizer_grouped_parameters += no_decay_param_group(trainable_params, lr, name=name)
            else:
                print(f"Module {name} has no trainable parameters, skipping...")

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        total_trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Total parameters: {len(list(self.parameters()))}")
        print(f"Trainable parameters: {len(total_trainable_parameters)}")
        print(f"Optimized parameters: {len(optimized_parameters)}")
        param
        assert len(optimized_parameters) == len(total_trainable_parameters), f"Some parameters are not optimized! Optimized: {len(optimized_parameters)}, Trainable: {len(total_trainable_parameters)}"
        return optimizer_grouped_parameters

    

@MODEL_REGISTRY.register()
class EmbodiedCRTInstSegModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories
        self.heads = cfg.model.heads
        self.inputs = ['mv', 'pts']
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        self.init_query_by_feat = cfg.model.get('init_query_by_feat', False)
        self.add_geometry_to_segment = cfg.model.get('add_geometry_to_segment', False)
        
        # build feature encoder
        for input in self.inputs:
            encoder = input + '_encoder'
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        # add geometry to segment
        if self.add_geometry_to_segment:
            self.pts_proj1 = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.feat_proj = nn.Sequential(
                nn.Linear(hidden_size + 3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        

        if self.init_query_by_feat:
            self.input_weights = nn.ParameterDict({
                input: nn.Parameter(torch.ones(1)) for input in self.memories
            })
         
    def forward(self, data_dict):
        input_dict = {}
        # build query
        mask = data_dict['query_pad_masks'].logical_not()
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        feat = torch.zeros_like(query_pos)
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center']
        fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'mv':
                # import pdb; pdb.set_trace()
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
                input_dict[input] = [feat, mask, pos]
            elif input == 'pts':
                voxel_features = data_dict['voxel_features']
                voxel_coordinates = data_dict['voxel_coordinates']
                x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
                voxel2segment = data_dict['voxel2segment']
         
                feat,pts_feat,pcds_w = self.pts_encoder(x, voxel2segment, max_seg=fts_locs.shape[1],data_dict=data_dict)
                if self.add_geometry_to_segment:#给segment注入对应点云的位置编码
                    for bid in range(len(voxel2segment)):
                        sp_idx = voxel2segment[bid]
                        all_xyz = data_dict['coordinates'][bid]
                        norm_xyz, _ = scatter_norm(all_xyz, sp_idx)
                        all_xyz_segment = scatter(self.pts_proj1(norm_xyz), sp_idx, dim=0, reduce='max', dim_size=fts_locs.shape[1])
                        for i in range(len(feat)):
                            feat[i][bid] = feat[i][bid] + all_xyz_segment
                    for i in range(len(feat)):        
                        feat[i] = self.feat_proj(torch.cat([feat[i], fts_locs], dim=-1)) 
                mask = data_dict['seg_pad_masks'].logical_not()
                data_dict['voxel_feat'] = {'feat': feat[-1].detach().cpu(), 'mask': mask.detach().cpu()}
                pos = fts_pos


                # import pudb; pudb.set_trace()  
                pts_mask = data_dict['voxel_pad_masks'].logical_not() #(20000,)
                voxel_to_full_maps = data_dict['voxel_to_full_maps']
                voxel_feat=pts_feat.decomposed_features
                
                pts_feat_list=[v_feat[idx] for v_feat,idx in zip(voxel_feat,voxel_to_full_maps)]
                pts2spidx = [v2s[v2p] for v2s, v2p in zip(data_dict['voxel2segment'], voxel_to_full_maps)]
              
                pts_feat_batched=torch.stack(pts_feat_list,dim=0).detach().cuda()

                pts_pos_batched=torch.from_numpy(np.stack([p[:,:3] for p in data_dict['raw_coordinates']],axis=0)).cuda()

                data_dict['pts_feat'] = {'feat': pts_feat_batched, 'mask': pts_mask.detach().cpu()}
    

                input_dict['voxel'] = [feat, mask, pos]
                input_dict['pts'] = [pts_feat_batched, pts_mask, pts_pos_batched]
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            
     
        # generate features for mask head
        seg_fts_for_match = []
        for input in self.inputs:
            if input in ['voxel', 'mv']:
                feats = copy(input_dict[input][:])
                if isinstance(feats[0], list):
                    assert input == 'voxel'
                    feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                seg_fts_for_match.append(feats)                 
     
        # generate features for spatial attention
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
        # init query
        if self.init_query_by_feat:
            input_query_feat_list = []
            for input in ['voxel','mv']:
                if input == 'voxel':
                    segment_feature = input_dict['voxel'][0][-1]
                else:
                    segment_feature = input_dict[input][0]
                query_selection_ids = data_dict['query_selection_ids']
                query_feat_list = []
                for bid in range(len(query_selection_ids)):
                    query_feat = segment_feature[bid][query_selection_ids[bid]]
                    query_feat_list.append(query_feat)
                query_feat = pad_sequence(query_feat_list, batch_first=True) * self.input_weights[input]
                input_query_feat_list.append(query_feat)
            query = torch.stack(input_query_feat_list, dim=-1).sum(dim=-1)
            assert query.shape == input_dict['query'][0].shape, f"Query shape {query.shape} does not match input shape {input_dict['query'][0].shape}"
            input_dict['query'] = (query, input_dict['query'][1], input_dict['query'][2])
        
    
        
        # unified encoding    
        # import pudb; pudb.set_trace()                       
        query, pred_ins_cls, predictions_class, predictions_mask, predictions_score, predictions_box = self.unified_encoder(input_dict, pairwise_locs, pcds_w, pts2spidx)
        data_dict['query_feat'] = query
        
        data_dict['predictions_score'] = [pred_ins_cls[-1]]
        data_dict['predictions_class'] = [predictions_class[-1]]
        data_dict['predictions_mask'] = [predictions_mask[-1]]
        data_dict['predictions_box'] = [predictions_box[-1]]
        # task head
        for head in self.heads:
            
            if head == 'openvocab':
                openvocab_query_feat = getattr(self, 'openvocab_head')(query)
                data_dict['openvocab_query_feat'] = openvocab_query_feat
            else:
                raise NotImplementedError(f"Unknow head type: {head}")
        # import pudb; pudb.set_trace()
        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            
            # 只收集需要梯度的参数
            trainable_params = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                optimizer_grouped_parameters += no_decay_param_group(trainable_params, lr, name=name)
            else:
                print(f"Module {name} has no trainable parameters, skipping...")

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        total_trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Total parameters: {len(list(self.parameters()))}")
        print(f"Trainable parameters: {len(total_trainable_parameters)}")
        print(f"Optimized parameters: {len(optimized_parameters)}")
        
        assert len(optimized_parameters) == len(total_trainable_parameters), f"Some parameters are not optimized! Optimized: {len(optimized_parameters)}, Trainable: {len(total_trainable_parameters)}"
        return optimizer_grouped_parameters


@MODEL_REGISTRY.register()
class EmbodiedCRTInstSegModelMV(EmbodiedCRTInstSegModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories
        self.heads = cfg.model.heads
        self.inputs = ['mv', 'pts']
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        self.init_query_by_feat = cfg.model.get('init_query_by_feat', False)
        self.add_geometry_to_segment = cfg.model.get('add_geometry_to_segment', False)
        
        # build feature encoder
        for input in self.inputs:
            encoder = input + '_encoder'
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        # add geometry to segment
        if self.add_geometry_to_segment:
            self.pts_proj1 = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.feat_proj = nn.Sequential(
                nn.Linear(hidden_size + 3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        

        if self.init_query_by_feat:
            self.input_weights = nn.ParameterDict({
                input: nn.Parameter(torch.ones(1)) for input in self.memories
            })
         
    def mv_forward(self, data_list_dict):
        num_frames=data_list_dict['total_frame_num']
        batch_list=data_list_dict['batch_list']

        for data_dict in batch_list: 
            input_dict = {}
            # build query
            mask = data_dict['query_pad_masks'].logical_not()
            query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
            coord_min = data_dict['coord_min']
            coord_max = data_dict['coord_max']
            query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
            feat = torch.zeros_like(query_pos)
            pos = query_pos
            input_dict['query'] = (feat, mask, pos)
            # encode fts including point, voxel, image, and prompt
            # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
            fts_locs = data_dict['seg_center']
            fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
            for input in self.inputs:
                feat, mask, pos = None, None, None
                if input == 'mv':
                    # import pdb; pdb.set_trace()
                    feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                    mask = data_dict['mv_seg_pad_masks'].logical_not()
                    pos = fts_pos
                    input_dict[input] = [feat, mask, pos]
                elif input == 'pts':
                    voxel_features = data_dict['voxel_features']
                    voxel_coordinates = data_dict['voxel_coordinates']
                    x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
                    voxel2segment = data_dict['voxel2segment']
            
                    feat,pts_feat,pcds_w = self.pts_encoder(x, voxel2segment, max_seg=fts_locs.shape[1],data_dict=data_dict)
                    if self.add_geometry_to_segment:#给segment注入对应点云的位置编码
                        for bid in range(len(voxel2segment)):
                            sp_idx = voxel2segment[bid]
                            all_xyz = data_dict['coordinates'][bid]
                            norm_xyz, _ = scatter_norm(all_xyz, sp_idx)
                            all_xyz_segment = scatter(self.pts_proj1(norm_xyz), sp_idx, dim=0, reduce='max', dim_size=fts_locs.shape[1])
                            for i in range(len(feat)):
                                feat[i][bid] = feat[i][bid] + all_xyz_segment
                        for i in range(len(feat)):        
                            feat[i] = self.feat_proj(torch.cat([feat[i], fts_locs], dim=-1)) 
                    mask = data_dict['seg_pad_masks'].logical_not()
                    data_dict['voxel_feat'] = {'feat': feat[-1].detach().cpu(), 'mask': mask.detach().cpu()}
                    pos = fts_pos


                    # import pudb; pudb.set_trace()  
                    pts_mask = data_dict['voxel_pad_masks'].logical_not() #(20000,)
                    voxel_to_full_maps = data_dict['voxel_to_full_maps']
                    voxel_feat=pts_feat.decomposed_features
                    
                    pts_feat_list=[v_feat[idx] for v_feat,idx in zip(voxel_feat,voxel_to_full_maps)]
                    pts2spidx = [v2s[v2p] for v2s, v2p in zip(data_dict['voxel2segment'], voxel_to_full_maps)]
                
                    pts_feat_batched=torch.stack(pts_feat_list,dim=0).detach().cuda()

                    pts_pos_batched=torch.from_numpy(np.stack([p[:,:3] for p in data_dict['raw_coordinates']],axis=0)).cuda()

                    data_dict['pts_feat'] = {'feat': pts_feat_batched, 'mask': pts_mask.detach().cpu()}
        

                    input_dict['voxel'] = [feat, mask, pos]
                    input_dict['pts'] = [pts_feat_batched, pts_mask, pts_pos_batched]
                else:
                    raise NotImplementedError(f"Unknow input type: {input}")
                
        
            # generate features for mask head
            seg_fts_for_match = []
            for input in self.inputs:
                if input in ['voxel', 'mv']:
                    feats = copy(input_dict[input][:])
                    if isinstance(feats[0], list):
                        assert input == 'voxel'
                        feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                    seg_fts_for_match.append(feats)                 
        
            # generate features for spatial attention
            if self.unified_encoder.spatial_selfattn:
                pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                            pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                            spatial_dim=self.spatial_dim)
            else:
                pairwise_locs = None
            # init query
            if self.init_query_by_feat:
                input_query_feat_list = []
                for input in ['voxel','mv']:
                    if input == 'voxel':
                        segment_feature = input_dict['voxel'][0][-1]
                    else:
                        segment_feature = input_dict[input][0]
                    query_selection_ids = data_dict['query_selection_ids']
                    query_feat_list = []
                    for bid in range(len(query_selection_ids)):
                        query_feat = segment_feature[bid][query_selection_ids[bid]]
                        query_feat_list.append(query_feat)
                    query_feat = pad_sequence(query_feat_list, batch_first=True) * self.input_weights[input]
                    input_query_feat_list.append(query_feat)
                query = torch.stack(input_query_feat_list, dim=-1).sum(dim=-1)
                assert query.shape == input_dict['query'][0].shape, f"Query shape {query.shape} does not match input shape {input_dict['query'][0].shape}"
                input_dict['query'] = (query, input_dict['query'][1], input_dict['query'][2])
            
        
            
            # unified encoding    
            # import pudb; pudb.set_trace()                       
            query, pred_ins_cls, predictions_class, predictions_mask, predictions_score, predictions_box = self.unified_encoder(input_dict, pairwise_locs, pcds_w, pts2spidx)
            data_dict['query_feat'] = query
            

            # task head
            for head in self.heads:
                if head == 'mask':
                    data_dict['predictions_score'] = [pred_ins_cls[-1]]
                    data_dict['predictions_class'] = [predictions_class[-1]]
                    data_dict['predictions_mask'] = [predictions_mask[-1]]
                    data_dict['predictions_box'] = [predictions_box[-1]]
                    continue
                elif head == 'openvocab':
                    openvocab_query_feat = getattr(self, 'openvocab_head')(query)
                    data_dict['openvocab_query_feat'] = openvocab_query_feat
                elif head == 'merge':
                    merge_query_feat = getattr(self, 'merge_head')(query)
                    data_dict['merge_query_feat'] = merge_query_feat
                else:
                    raise NotImplementedError(f"Unknow head type: {head}")
            
        
        # import pudb; pudb.set_trace()
        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            
            # 只收集需要梯度的参数
            trainable_params = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                optimizer_grouped_parameters += no_decay_param_group(trainable_params, lr, name=name)
            else:
                print(f"Module {name} has no trainable parameters, skipping...")

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        total_trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Total parameters: {len(list(self.parameters()))}")
        print(f"Trainable parameters: {len(total_trainable_parameters)}")
        print(f"Optimized parameters: {len(optimized_parameters)}")
        
        assert len(optimized_parameters) == len(total_trainable_parameters), f"Some parameters are not optimized! Optimized: {len(optimized_parameters)}, Trainable: {len(total_trainable_parameters)}"
        return optimizer_grouped_parameters





import spconv.pytorch as spconv

@MODEL_REGISTRY.register()
class EmbodiedPQ3DInstSegModelSPconv(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories
        self.heads = cfg.model.heads
        self.inputs = self.memories[:]
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        self.init_query_by_feat = cfg.model.get('init_query_by_feat', False)
        self.add_geometry_to_segment = cfg.model.get('add_geometry_to_segment', False)
        
        # build feature encoder
        for input in self.inputs:
            encoder = input + '_encoder'
            setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        # add geometry to segment
        if self.add_geometry_to_segment:
            self.pts_proj1 = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.feat_proj = nn.Sequential(
                nn.Linear(hidden_size + 3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        if self.init_query_by_feat:
            self.input_weights = nn.ParameterDict({
                input: nn.Parameter(torch.ones(1)) for input in self.inputs
            })
         
    def forward(self, data_dict):
        input_dict = {}
        # build query
        mask = data_dict['query_pad_masks'].logical_not()
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        feat = torch.zeros_like(query_pos)
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center']
        fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'mv':
                # import pdb; pdb.set_trace()
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'voxel':
                voxel_features = data_dict['voxel_features']
                voxel_coordinates = data_dict['voxel_coordinates']                
                voxel2segment = data_dict['voxel2segment']
                sonata_pts=data_dict['sonata_pts']
                feat = self.voxel_encoder(sonata_pts, voxel2segment, max_seg=fts_locs.shape[1])
                if self.add_geometry_to_segment:
                    for bid in range(len(voxel2segment)):
                        sp_idx = voxel2segment[bid]
                        all_xyz = data_dict['coordinates'][bid]
                        norm_xyz, _ = scatter_norm(all_xyz, sp_idx)
                        all_xyz_segment = scatter(self.pts_proj1(norm_xyz), sp_idx, dim=0, reduce='max', dim_size=fts_locs.shape[1])
                        for i in range(len(feat)):
                            feat[i][bid] = feat[i][bid] + all_xyz_segment
                    for i in range(len(feat)):        
                        feat[i] = self.feat_proj(torch.cat([feat[i], fts_locs], dim=-1)) 
                mask = data_dict['seg_pad_masks'].logical_not()
                data_dict['voxel_feat'] = {'feat': feat[-1].detach().cpu(), 'mask': mask.detach().cpu()}
                pos = fts_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            input_dict[input] = [feat, mask, pos]
        # build offline attention mask for guided mask training
        offline_attn_masks = None
        # generate features for mask head
        seg_fts_for_match = []
        for input in self.inputs:
            if input in ['voxel', 'mv']:
                feats = copy(input_dict[input][:])
                if isinstance(feats[0], list):
                    assert input == 'voxel'
                    feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                seg_fts_for_match.append(feats)                 
        # import pdb; pdb.set_trace()
        # build mask head
        if hasattr(self, 'mask_head'):
            mask_head_partial = partial(self.mask_head, query_locs=query_locs, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                        offline_attn_masks=offline_attn_masks, skip_prediction=self.skip_query_encoder_mask_pred)
        else:
            mask_head_partial = None
        # generate features for spatial attention
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
        # init query
        if self.init_query_by_feat:
            input_query_feat_list = []
            for input in self.inputs:
                if input == 'voxel':
                    segment_feature = input_dict['voxel'][0][-1]
                else:
                    segment_feature = input_dict[input][0]
                query_selection_ids = data_dict['query_selection_ids']
                query_feat_list = []
                for bid in range(len(query_selection_ids)):
                    query_feat = segment_feature[bid][query_selection_ids[bid]]
                    query_feat_list.append(query_feat)
                query_feat = pad_sequence(query_feat_list, batch_first=True) * self.input_weights[input]
                input_query_feat_list.append(query_feat)
            query = torch.stack(input_query_feat_list, dim=-1).sum(dim=-1)
            assert query.shape == input_dict['query'][0].shape, f"Query shape {query.shape} does not match input shape {input_dict['query'][0].shape}"
            input_dict['query'] = (query, input_dict['query'][1], input_dict['query'][2])
        # unified encoding                           
        query, predictions_score, predictions_class, predictions_mask, predictions_box = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)
        data_dict['query_feat'] = query
        
        # task head
        for head in self.heads:
            if head == 'mask':
                if self.skip_query_encoder_mask_pred:
                    mask_head_partial = partial(self.mask_head, query_locs=query_locs, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                    offline_attn_masks=offline_attn_masks, skip_prediction=False)
                    predictions_score = []
                    predictions_class = []
                    predictions_mask = []
                    predictions_box = []
                pred_scores, pred_logits, pred_masks, pred_boxes, _ = mask_head_partial(query=query)
                predictions_score.append(pred_scores)
                predictions_class.append(pred_logits)
                predictions_mask.append(pred_masks)
                predictions_box.append(pred_boxes)
                data_dict['predictions_score'] = predictions_score
                data_dict['predictions_class'] = predictions_class
                data_dict['predictions_mask'] = predictions_mask
                data_dict['predictions_box'] = predictions_box  
                continue
            elif head == 'openvocab':
                openvocab_query_feat = getattr(self, 'openvocab_head')(query)
                data_dict['openvocab_query_feat'] = openvocab_query_feat
            else:
                raise NotImplementedError(f"Unknow head type: {head}")
       
        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            
            # 只收集需要梯度的参数
            trainable_params = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                optimizer_grouped_parameters += no_decay_param_group(trainable_params, lr, name=name)
            else:
                print(f"Module {name} has no trainable parameters, skipping...")

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        total_trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Total parameters: {len(list(self.parameters()))}")
        print(f"Trainable parameters: {len(total_trainable_parameters)}")
        print(f"Optimized parameters: {len(optimized_parameters)}")
        
        assert len(optimized_parameters) == len(total_trainable_parameters), f"Some parameters are not optimized! Optimized: {len(optimized_parameters)}, Trainable: {len(total_trainable_parameters)}"
        return optimizer_grouped_parameters
