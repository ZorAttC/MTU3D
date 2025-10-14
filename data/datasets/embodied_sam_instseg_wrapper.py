import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer
import MinkowskiEngine as ME

from data.datasets.constant import PromptType

from .dataset_wrapper import DATASETWRAPPER_REGISTRY
from ..data_utils import pad_sequence, pad_sequence_2d



@DATASETWRAPPER_REGISTRY.register()
class EmbodiedSAMInstSegMVDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.offline_mask_source = self.dataset.offline_mask_source
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, batch_list):
        # batch list [[dict01, dict02], [dict11, dict12, ..], [...]] 
        res_batch_list = []
        for i in range(len(batch_list[0])):
            batch = [sample[i] for sample in batch_list]
            new_batch = {}

            # sparse collate voxel features
            input_dict = {
                "coords": [sample.pop('voxel_coordinates') for sample in batch], 
                "feats": [sample.pop('voxel_features') for sample in batch],
            }
            voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
            new_batch['voxel_coordinates'] = voxel_coordinates
            new_batch['voxel_features'] = voxel_features
            
            # load offline mask
            if self.offline_mask_source is not None:
                # build the gt attn mask from the gt segment masks. True as masked.
                if self.offline_mask_source == 'gt':
                    padded_segment_masks, padding_mask = pad_sequence_2d([sample['segment_masks'] for sample in batch], return_mask=True)
                    new_batch['offline_attn_mask'] = padded_segment_masks.logical_not()
                    
                    # build labels and masks for loss
                    labels = pad_sequence([sample['instance_labels'] for sample in batch], pad=-100)
                    new_batch['target_labels'] = labels
                    new_batch['target_masks'] = padded_segment_masks.float()
                    new_batch['target_masks_pad_masks'] = padding_mask.logical_not()
                else: 
                    raise NotImplementedError(f'{self.offline_mask_source} is not implemented')
                
            # list collate
            list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'instance_boxes', 'instance_ids_ori', 'full_masks', 'segment_masks', 'scan_id', 'segment_labels', 'query_selection_ids']
            for k in list_keys:
                new_batch[k] = [sample.pop(k) for sample in batch]

            # pad collate: ['coord_min', 'coord_max', 'seg_center', 'seg_pad_mask', 'obj_center', 'obj_pad_mask']
            padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'seg_point_count', 'query_locs', 'query_pad_masks',
                            'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks', 'pc_seg_fts', 'pc_seg_pad_masks', 'prompt', 'prompt_pad_masks']
            padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
            for k in padding_keys:
                tensors = [sample.pop(k) for sample in batch]
                padded_tensor = pad_sequence(tensors)
                new_batch[k] = padded_tensor
        
            # default collate
            new_batch.update(default_collate(batch))
            
            # add to batch list
            res_batch_list.append(new_batch)
            
        return {'batch_list': res_batch_list, 'frame_id': 0, 'total_frame_num': len(res_batch_list)}
