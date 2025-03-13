import json
import os
from pathlib import Path
import random
import jsonlines
import numpy as np
import torch
import volumentations as V
import albumentations as A

from data.build import DATASET_REGISTRY
from data.datasets.constant import PromptType, CLASS_LABELS_REPLICA
from torch.utils.data import Dataset

from copy import deepcopy


@DATASET_REGISTRY.register()
class ReplicaPromptInstSeg(Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        split = split if split != 'test' else 'val'
        scan_ids = ["office0", "office1", "office2", "office3", "room0", "room1", "room2"]
        self.load_offline_segment_voxel = cfg.data.get('load_offline_segment_voxel', False)
        self.load_offline_segment_image = cfg.data.get('load_offline_segment_image', False)
        self.load_offline_segment_point = cfg.data.get('load_offline_segment_point', False)
        self.base_dir = cfg.data.get('replica_base')
        self.scans = self._load_replica(scan_ids=scan_ids, load_offline_segment_voxel=self.load_offline_segment_voxel, load_offline_segment_image=self.load_offline_segment_image, load_offline_segment_point=self.load_offline_segment_point)
        self.scans_keys = scan_ids
        self.ignore_label = cfg.data.ignore_label
        # load augmentations
        self.volume_augmentations = V.NoOp()
        self.image_augmentations = A.NoOp()
        if cfg.data.volume_augmentations_path is not None:
            self.volume_augmentations = V.load(
                Path(cfg.data.volume_augmentations_path), data_format="yaml"
            )
        if cfg.data.image_augmentations_path is not None:
            self.image_augmentations = A.load(
                Path(cfg.data.image_augmentations_path), data_format="yaml"
            )
        color_mean = [0.47793125906962, 0.4303257521323044, 0.3749598901421883]
        color_std = [0.2834475483823543, 0.27566157565723015, 0.27018971370874995]
        self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
        self.use_aug = cfg.data.get('use_aug', True)
        # load langauge
        self.sources = cfg.data.get(f'{split}_sources', [])
        self.prompt_data, self.scan_ids = self._load_prompt(cfg.data, scan_ids)
        
    def __len__(self):
        return len(self.prompt_data)
    
    def __getitem__(self, index):
        # load lang
        item = self.prompt_data[index]
        scan_id =  item['scan_id']
        tgt_object_id = item['target_id']
        prompt = item['utterance']
        prompt_type = item['prompt_type']
        # load points
        points = deepcopy(self.scans[scan_id]['scene_pcds']) # mask3d points
        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )
        color = (color + 1) * 127.5
        
        raw_coordinates = coordinates.copy()
        
        if self.split == 'train' and self.use_aug:
            # random augmentations
            coordinates -= coordinates.mean(0)
            coordinates += (
                np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            )
            
            # revser x y axis
            for i in (0, 1):
                if random.random() < 0.5:
                    coord_max = np.max(coordinates[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]
                    
            # volume augmentation
            aug = self.volume_augmentations(
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            # color augmentation
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )
            
        # normalize color
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        
        # prepare labels and map from 0 to 199 (-100 for ignore label)
        labels = labels.astype(np.int32)
        labels = np.hstack((labels, segments[..., None].astype(np.int32)))
        
        # prepare features, Nx6
        features = np.hstack((color, coordinates))
        
        # data_dict
        data_dict = {
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "raw_coordinates": raw_coordinates,
            "idx": index,
            # for refer
            "prompt": prompt,
            "prompt_type": prompt_type,
            "tgt_object_id": torch.LongTensor([tgt_object_id] if type(tgt_object_id) != list else tgt_object_id),
        }
        
        if self.load_offline_segment_voxel:
            cur_segment_voxel = deepcopy(self.scans[scan_id]['offline_segment_voxel'])
            data_dict.update(cur_segment_voxel)
        
        if self.load_offline_segment_image:
            cur_segment_image = deepcopy(self.scans[scan_id]['offline_segment_image'])
            data_dict.update(cur_segment_image)
        
        if self.load_offline_segment_point:
            cur_segment_point = deepcopy(self.scans[scan_id]['offline_segment_point'])
            data_dict.update(cur_segment_point)
        
        return data_dict
        
    def _load_replica(self, scan_ids, load_offline_segment_voxel=False, load_offline_segment_image=False, load_offline_segment_point=False):
        scans = {}
        for scan_id in scan_ids:
            one_scan = {}
            one_scan['scene_pcds'] = np.load(os.path.join(self.base_dir, "pcd_mask3d", f"{scan_id}.npy"))
            
            if load_offline_segment_voxel:
                pass
            
            if load_offline_segment_image:
                one_scan['offline_segment_image'] = torch.load(os.path.join(self.base_dir, "mask3d_image_feature", f'{scan_id}.pth'))
            
            if load_offline_segment_point:
                one_scan['offline_segment_point'] = torch.load(os.path.join(self.base_dir, "mask3d_point_feature", f'{scan_id}.pth'))
            
            scans[scan_id] = one_scan
        return scans
    
    def _load_prompt(self, cfg, split_scan_ids=None):
        prompt_data = []
        sources = self.sources
        scan_ids = set()

        if sources:            
            if 'prompt_all_class' in sources:
                prompt_all = ""
                for class_name in CLASS_LABELS_REPLICA:
                    if class_name not in ['wall', 'floor']:
                        prompt_all += ":" + class_name
                      
                for scan_id in split_scan_ids:
                    scan_ids.add(scan_id)
                    item = {}
                    item['scan_id'] = scan_id
                    item['utterance'] = ""
                    item['target_id'] = [-1]
                    item['instance_type'] = prompt_all
                    item['prompt_type'] = PromptType.TEXT
                    prompt_data.append(item)
                     
        return prompt_data, scan_ids
