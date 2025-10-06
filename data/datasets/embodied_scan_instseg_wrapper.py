import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer
import MinkowskiEngine as ME

from data.datasets.constant import PromptType

from .dataset_wrapper import DATASETWRAPPER_REGISTRY
from ..data_utils import pad_sequence, pad_sequence_2d
import modules.third_party.sonata as sonata

@DATASETWRAPPER_REGISTRY.register()
class EmbodiedScanInstSegDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        new_batch = {}

        # sparse collate voxel features
        input_dict = {
            "coords": [sample.pop('voxel_coordinates') for sample in batch], 
            "feats": [sample.pop('voxel_features') for sample in batch],
        }
        voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
        new_batch['voxel_coordinates'] = voxel_coordinates
        new_batch['voxel_features'] = voxel_features
            
        # list collate
        list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'instance_boxes', 'instance_ids_ori', 'full_masks', 'segment_masks', 'scan_id', 'segment_labels', 'query_selection_ids', 'instance_text_labels', 'instance_text_embeds']
        list_keys = [k for k in list_keys if k in batch[0].keys()]
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
            
        # pad collate
        padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'seg_point_count', 'query_locs', 'query_pad_masks',
                        'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks']
        padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padded_tensor = pad_sequence(tensors)
            new_batch[k] = padded_tensor
        
        # default collate
        new_batch.update(default_collate(batch))
        return new_batch

@DATASETWRAPPER_REGISTRY.register()
class EmbodiedScanInstSegDatasetWrapperSPconv(Dataset):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        new_batch = {}

        # sparse collate voxel features
        input_dict = {
            "coords": [sample.pop('voxel_coordinates') for sample in batch], 
            "feats": [sample.pop('voxel_features') for sample in batch],
        }
        voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
        
        # sonata data collate
        sonata_pts_list=[sample.pop('sonata_pts') for sample in batch]
        batch_sonata_pts=sonata.data.collate_fn(sonata_pts_list)
        new_batch['sonata_pts'] = batch_sonata_pts
        
        new_batch['voxel_coordinates'] = voxel_coordinates
        new_batch['voxel_features'] = voxel_features
            
        # list collate
        list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'instance_boxes', 'instance_ids_ori', 'full_masks', 'segment_masks', 'scan_id', 'segment_labels', 'query_selection_ids', 'instance_text_labels', 'instance_text_embeds']
        list_keys = [k for k in list_keys if k in batch[0].keys()]
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
            
        # pad collate
        padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'seg_point_count', 'query_locs', 'query_pad_masks',
                        'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks']
        padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padded_tensor = pad_sequence(tensors)
            new_batch[k] = padded_tensor
        
        # default collate
        new_batch.update(default_collate(batch))
        return new_batch