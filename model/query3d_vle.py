from copy import copy
from functools import partial
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from data.data_utils import pad_sequence
from data.datasets.constant import PromptType

from modules.build import build_module_by_name
from modules.utils import calc_pairwise_locs
from model.build import MODEL_REGISTRY, BaseModel
from optim.utils import no_decay_param_group
from model.mask3d import CoordinateEncoder
        
@MODEL_REGISTRY.register()
class Query3DVLE(BaseModel):
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
        # build prompt type
        self.prompt_types = ['txt', 'loc', 'image']
        # build feature encoder
        for input in self.inputs:
            if input == 'prompt':
                for prompt_type in self.prompt_types: # only text prompt for now
                    if prompt_type == 'loc':
                        continue
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
            else:
                encoder = input + '_encoder'
                setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.box_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        # build obj frontier embedding
        self.frontier_embedding = nn.Embedding(2, self.hidden_size)
        
    def prompt_encoder(self, data_dict):
        prompt = data_dict['prompt']
        prompt_pad_masks = data_dict['prompt_pad_masks']
        prompt_type = data_dict['prompt_type']
        prompt_feat = torch.zeros(prompt_pad_masks.shape + (self.hidden_size,), device=prompt_pad_masks.device)
        for type in self.prompt_types:
            # get idx
            idx = prompt_type == getattr(PromptType, type.upper())
            if idx.sum() == 0:
                continue
            input = []
            for i in range(len(prompt)):
                if idx[i]:
                    input.append(prompt[i])
            mask = prompt_pad_masks[idx]
            # encode
            if type == 'txt':
                input = pad_sequence(input, pad=0)
                encoder = self.txt_encoder
                feat = encoder(input.long(), mask)
            elif type == 'loc':
                loc_prompts = input[:, :self.dim_loc]
                if self.dim_loc > 3:
                    feat = self.coord_encoder(loc_prompts[:, :3]).unsqueeze(1) + self.box_encoder(loc_prompts[:, 3:6]).unsqueeze(1)
                else:
                    feat = self.coord_encoder(loc_prompts[:, :3].unsqueeze(1), input_range=[data_dict['coord_min'][idx], data_dict['coord_max'][idx]])
                mask[:, 1:] = False
            elif type == 'image':
                img_prompts = torch.stack(input).unsqueeze(1)
                feat = self.image_encoder(img_prompts)
                mask[:, 1:] = False
            else:
                raise NotImplementedError(f'{type} is not implemented')
            # put back to orignal prompt
            prompt_feat[idx] = feat
            prompt_pad_masks[idx] = mask
        return prompt_feat, prompt_pad_masks.logical_not()
        
    def forward(self, data_dict):
        input_dict = {}
        # build query
        mask = data_dict['query_pad_masks'].logical_not()
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        query_pos  = self.coord_encoder(query_locs[:, :, :3]) + self.box_encoder(query_locs[:, :, 3:6])
        feat = torch.zeros_like(query_pos)
        real_obj_pad_masks = data_dict['real_obj_pad_masks']
        frontier_emb = self.frontier_embedding(real_obj_pad_masks.long())
        feat += frontier_emb
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center']
        fts_pos = self.coord_encoder(fts_locs[:, :, :3]) + self.box_encoder(fts_locs[:, :,  3:6])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'prompt':
                feat, mask = self.prompt_encoder(data_dict)
            elif input == 'mv':
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'vocab':
                feat = self.vocab_encoder(data_dict['vocab_seg_fts'])
                mask = data_dict['vocab_seg_pad_masks'].logical_not()
                pos = fts_pos
            else:
                raise NotImplementedError(f'{input} is not implemented')
            input_dict[input] = [feat, mask, pos]
        mask_head_partial = None
        # generate features for spatial attention
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
            
        # unified encoding                           
        query, predictions_score, predictions_class, predictions_mask, predictions_box = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)
        
        # task head
        for head in self.heads:
            if head == 'ground':
                inputs = [query, data_dict['query_pad_masks']]
                label = data_dict["tgt_object_id"]
                logits = getattr(self, head + '_head')(*inputs)
                data_dict[head + '_logits'] = logits
                data_dict['og3d_logits'] = logits
                data_dict[head + '_label'] = label
            elif head == 'query_cls':
                label = data_dict["obj_labels"]
                logits = getattr(self, head + '_head')(query)
                data_dict[head + '_logits'] = logits
                data_dict[head + '_label'] = label
            elif head == 'decision':
                label = data_dict['decision_label']
                logits = getattr(self, head + '_head')(query, data_dict['query_pad_masks'])
                data_dict[head + '_logits'] = logits
                data_dict[head + '_label'] = label
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
            optimizer_grouped_parameters += no_decay_param_group(module.named_parameters(), lr, name=name)

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        assert len(optimized_parameters) == len(list(self.parameters())), "Some parameters are not optimized!"
        return optimizer_grouped_parameters
        