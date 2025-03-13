import json
import os
from pathlib import Path
from copy import deepcopy
import random
import jsonlines
import numpy as np
import torch
from torch_scatter import scatter_mean
from transformers import AutoTokenizer
import volumentations as V
import albumentations as A
import MinkowskiEngine as ME

from data.build import DATASET_REGISTRY
from data.datasets.scannet_base import ScanNetBase
from data.datasets.sceneverse_base import SceneVerseBase
from data.datasets.constant import CLASS_LABELS_200, PromptType
from data.data_utils import make_bce_label
import fpsample

from data.datasets.sceneverse_instseg import SceneVerseInstSeg

@DATASET_REGISTRY.register()
class ScanNetInstSeg(ScanNetBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        # basic params
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'
        # load scan data
        self.scan_ids = self._load_split(self.cfg, self.split)
        if self.split == 'train' and dataset_cfg.get('include_val_for_train', False):
            self.scan_ids += self._load_split(self.cfg, 'val')
        self.init_scan_data()
        # data process params
        self.ignore_label = dataset_cfg.ignore_label
        self.filter_out_classes = dataset_cfg.filter_out_classes
        self.voxel_size = dataset_cfg.voxel_size
        self.use_aug = dataset_cfg.get('use_aug', True)
        # query init params
        self.num_queries = dataset_cfg.get('num_queries', 120)
        self.query_sample_strategy = dataset_cfg.get('query_sample_strategy', 'fps')
        self.offline_mask_source = dataset_cfg.get('offline_mask_source', None)
        assert self.query_sample_strategy in ['fps', 'gt']

        # extract instance info from scene_pcds
        self.extract_inst_info()
        
        # load augmentations
        self.volume_augmentations = V.NoOp()
        self.image_augmentations = A.NoOp()
        if dataset_cfg.volume_augmentations_path is not None:
            self.volume_augmentations = V.load(
                Path(dataset_cfg.volume_augmentations_path), data_format="yaml"
            )
        if dataset_cfg.image_augmentations_path is not None:
            self.image_augmentations = A.load(
                Path(dataset_cfg.image_augmentations_path), data_format="yaml"
            )
        color_mean = [0.47793125906962, 0.4303257521323044, 0.3749598901421883]
        color_std = [0.2834475483823543, 0.27566157565723015, 0.27018971370874995]
        self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
        
    def __len__(self):
        return len(self.scan_ids)
    
    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        data_dict = self.get_scene(scan_id)
        return data_dict

    def extract_inst_info(self):
        for scan_id in self.scan_ids:
            scan_data = self.scan_data[scan_id]
            labels = scan_data['scene_pcds'][:, 9:12]
            point2seg_id, point2inst_label, point2inst_id = labels[:, 0], labels[:, 1], labels[:, 2]

            # map instance labels to scannet200 id, modify scan_data in-place
            point2inst_label = self.map_to_scannet200_id(point2inst_label)

            point2seg_id = point2seg_id.astype(int)
            point2inst_label = point2inst_label.astype(int)
            point2inst_id = point2inst_id.astype(int)

            # get unique instances
            unique_inst_ids, indices, inverse_indices = np.unique(point2inst_id, return_index=True, return_inverse=True)
            unique_inst_labels = point2inst_label[indices]
            n_inst = len(unique_inst_ids)

            # instance to point mask
            instance_range = np.arange(n_inst)[:, None]
            full_masks = instance_range == inverse_indices

            # instance to segment mask
            n_segments = point2seg_id.max() + 1
            segment_masks = np.zeros((n_inst, n_segments), dtype=bool)
            segment_masks[inverse_indices, point2seg_id] = True

            # filter out unwanted instances
            valid = (unique_inst_ids != -1) & ~np.isin(unique_inst_labels, self.filter_out_classes)
            unique_inst_ids = unique_inst_ids[valid]
            unique_inst_labels = unique_inst_labels[valid]
            full_masks = full_masks[valid]
            segment_masks = segment_masks[valid]
            
            inst_info = {
                'instance_ids': torch.LongTensor(unique_inst_ids),
                'instance_labels': torch.LongTensor(unique_inst_labels),
                'full_masks': torch.LongTensor(full_masks),
                'segment_masks': torch.LongTensor(segment_masks),
            }
            scan_data['inst_info'] = inst_info
    
    def sample_query(self, voxel_coordinates, coordinates, obj_center):
        if self.query_sample_strategy == 'fps':
            fps_idx = torch.from_numpy(fpsample.bucket_fps_kdline_sampling(voxel_coordinates.numpy(), self.num_queries, h=3).astype(np.int32))
            sampled_coords = coordinates[fps_idx]
            return sampled_coords, torch.ones(len(sampled_coords), dtype=torch.bool)
        elif self.query_sample_strategy == 'gt':
            return obj_center, torch.ones(len(obj_center), dtype=torch.bool)
        else:
            raise NotImplementedError(f'{self.query_sample_strategy} is not implemented')

    def get_scene(self, scan_id):
        points = deepcopy(self.scan_data[scan_id]['scene_pcds']) # mask3d points
        coordinates, color, normals, point2seg_id, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12]  if points.shape[1] > 10 else np.ones((points.shape[0], 2)) * 2,
        )
        
        if self.split == 'train' and self.use_aug:
            # random augmentations
            coordinates -= coordinates.mean(0)
            coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            
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
        features = np.hstack((color, coordinates))

        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()
        coordinates = torch.from_numpy(coordinates).float()

        # Calculate object centers and segment centers
        instance_info = deepcopy(self.scan_data[scan_id]['inst_info'])
        instance_ids = instance_info['instance_ids']
        point2inst_id = labels[:, -1]
        point2inst_id[point2inst_id == -1] = instance_ids.max() + 1
        obj_center = scatter_mean(coordinates, point2inst_id, dim=0)
        obj_center = obj_center[instance_ids]
        point2seg_id = torch.from_numpy(point2seg_id).long()
        seg_center = scatter_mean(coordinates, point2seg_id, dim=0)

        # voxelize coorinates, features and labels
        voxel_coordinates = np.floor(coordinates / self.voxel_size)
        _, unique_map, inverse_map = ME.utils.sparse_quantize(coordinates=voxel_coordinates, return_index=True, return_inverse=True)
        voxel_coordinates = voxel_coordinates[unique_map]
        voxel_features = features[unique_map]
        voxel2seg_id = point2seg_id[unique_map]
        
        # sample queries
        query_locs, query_pad_masks = self.sample_query(voxel_coordinates, coordinates, obj_center)
        
        # fill data dict
        data_dict = {
            # input data
            "voxel_coordinates": voxel_coordinates,
            "voxel_features": voxel_features,
            "voxel2segment": voxel2seg_id,
            "coordinates": voxel_features[:, -3:],
            # full, voxel, segment mappings
            "voxel_to_full_maps": inverse_map,
            "segment_to_full_maps": point2seg_id,
            # for computing iou in raw data
            "raw_coordinates": coordinates.numpy(),
            "coord_min": coordinates.min(0)[0],
            "coord_max": coordinates.max(0)[0],
            "scan_id": scan_id,
            # extra instance info
            "obj_center": obj_center, 
            "obj_pad_masks": torch.ones(len(obj_center), dtype=torch.bool),
            "seg_center": seg_center,
            "seg_pad_masks": torch.ones(len(seg_center), dtype=torch.bool),
            # query info
            'query_locs': query_locs,
            'query_pad_masks': query_pad_masks
        }
        data_dict.update(instance_info)
        
        if 'offline_segment_voxel' in self.scan_data[scan_id]:
            cur_segment_voxel = deepcopy(self.scan_data[scan_id]['offline_segment_voxel'])
            data_dict['voxel_seg_fts'] = cur_segment_voxel['voxel_seg_feature']
            data_dict['voxel_seg_pad_masks'] = data_dict['seg_pad_masks'].clone()
            assert data_dict['voxel_seg_fts'][0].shape[1] == len(data_dict['seg_center'])
        
        if 'offline_segment_image' in self.scan_data[scan_id]:
            cur_segment_image = deepcopy(self.scan_data[scan_id]['offline_segment_image'])
            data_dict['mv_seg_fts'] = cur_segment_image['image_seg_feature']
            data_dict['mv_seg_pad_masks'] = cur_segment_image['image_seg_mask'].logical_not()
            assert len(data_dict['mv_seg_fts']) == len(data_dict['seg_center'])
        
        if 'offline_segment_point' in self.scan_data[scan_id]:
            cur_segment_point = deepcopy(self.scan_data[scan_id]['offline_segment_point'])
            data_dict['pc_seg_fts'] = cur_segment_point['point_seg_feature']
            data_dict['pc_seg_pad_masks'] = cur_segment_point['point_seg_mask'].logical_not()
            assert len(data_dict['pc_seg_fts']) == len(data_dict['seg_center'])
        
        return data_dict
        
    def map_to_scannet200_id(self, labels):
        label_info = self.label_converter.scannet_raw_id_to_scannet200_id
        labels[~np.isin(labels, list(label_info))] = self.ignore_label
        for k in label_info:
            labels[labels == k] = label_info[k]
        return labels

@DATASET_REGISTRY.register()
class ScanNetInstSegSceneVerse(SceneVerseInstSeg):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)  
          

@DATASET_REGISTRY.register()
class ScanNetPromptInstSeg(ScanNetInstSeg):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        
        # load langauge
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        tokenizer_name = dataset_cfg.get('tokenizer', 'openai/clip-vit-large-patch14')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.prompt_nonexist_ratio = dataset_cfg.get('prompt_nonexist_ratio', 0.0)
        self.sources = dataset_cfg.get(f'{split}_sources', ['prompt_all_class'])
        self.lang_data = self._load_lang()
        self.inst_label2txt_ids = self.tokenize_class_names()
        
    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        scan_id, data_dict = self.get_lang(index)
        scene_dict = self.get_scene(scan_id)
        data_dict.update(scene_dict)
        return data_dict
    
    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scan_id']
        inst_info = self.scan_data[scan_id]['inst_info']
        instance_labels = inst_info['instance_labels']

        if random.random() < self.prompt_nonexist_ratio:
            nonexist_labels = set(range(len(CLASS_LABELS_200))) - set(instance_labels.numpy().tolist() + self.filter_out_classes)
            nonexist_label = random.choice(list(nonexist_labels))
            txt_ids = self.inst_label2txt_ids[nonexist_label]
            target_id = []
        else:
            refer_data = random.choice(item['refer_data'])
            txt_ids = self.inst_label2txt_ids[refer_data['instance_label']]
            target_id = refer_data['target_id']
        
        # remap target_id to tgt_object_id
        instance_ids = inst_info['instance_ids'].numpy().tolist()
        tgt_object_id = [instance_ids.index(inst_id) for inst_id in target_id]
        tgt_object_id = make_bce_label(tgt_object_id, num_classes=len(instance_ids))

        data_dict = {
            "prompt": torch.LongTensor(txt_ids),
            "prompt_pad_masks": torch.ones(len(txt_ids)).bool(),
            "prompt_type": PromptType.TXT,
            "tgt_object_id": tgt_object_id,
        }

        return scan_id, data_dict

    def _load_lang(self):
        split_scan_ids = self.scan_ids
        lang_data = []
        sources = self.sources

        if 'scanrefer' in sources:
            anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
            with jsonlines.open(anno_file, 'r') as _f:
                for item in _f:
                    if item['scan_id'] in split_scan_ids:
                        lang_data.append(item)
                        
        if self.split == 'train':
            # use class name as the sentence
            for scan_id in split_scan_ids:
                item = {}
                item['scan_id'] = scan_id
                inst_info = self.scan_data[scan_id]['inst_info']
                instance_ids = inst_info['instance_ids'].numpy().tolist()
                instance_labels = inst_info['instance_labels'].numpy().tolist()
                refer_data = []
                for inst_label in set(instance_labels):
                    data = {}
                    class_name = CLASS_LABELS_200[inst_label]
                    data['instance_label'] = inst_label
                    data['target_id'] = [id for id, label in zip(instance_ids, instance_labels) if label == inst_label]
                    data['instance_type'] = class_name
                    refer_data.append(data)
                item['refer_data'] = refer_data
                lang_data.append(item)
        else:
            all_class = [c for c in CLASS_LABELS_200 if c not in ['wall', 'floor']]
            all_class = ':'.join(all_class)
                    
            for scan_id in split_scan_ids:
                item = {}
                item['scan_id'] = scan_id
                item['refer_data'] = [{'instance_label': 0, 'target_id': [], 'instance_type': all_class}]
                lang_data.append(item)
                            
        return lang_data
    
    def tokenize_class_names(self):
        sentences = [f"a {class_name} in a scene" for class_name in CLASS_LABELS_200]
        txt_ids = self.tokenizer(sentences, add_special_tokens=True, truncation=True).input_ids
        return txt_ids
