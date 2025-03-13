import os
import collections
import json
import random
import jsonlines
import torch

from ..build import DATASET_REGISTRY
from ..data_utils import convert_pc_to_box, construct_bbox_corners, \
                         eval_ref_one_sample, is_explicitly_view_dependent
from .scannet_base import ScanNetBase
from .data_augmentor import DataAugmentor


@DATASET_REGISTRY.register()
class ScanNetPretrain(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetPretrain, self).__init__(cfg, split)
        self.pc_type = cfg.data.args.pc_type
        self.max_obj_len = cfg.data.args.max_obj_len
        self.num_points = cfg.data.args.num_points
        # TODO: only scanrefer needs test set
        if split != 'train':
            split = 'val'
        self.scannet_scan_ids = self._load_split(cfg, split)
        self.rot_aug = cfg.data.args.rot_aug
        self.aug_cfg = getattr(cfg, 'data_aug', None)
        if self.aug_cfg:
            self.augmentor = DataAugmentor(self.aug_cfg, self.split)

        split_cfg = cfg.data.get(self.__class__.__name__).get(split)

        print(f"Loading ScanNet {split}-set language")
        self.lang_data = self._load_lang(split_cfg)
        print(f"Finish loading ScanNet {split}-set language of size {self.__len__()}")

        print(f"Loading ScanNet {split}-set scans")
        self.scan_data = self._load_scannet(self.scannet_scan_ids, self.pc_type,
                                           load_inst_info = True)
        print(f"Finish loading ScanNet {split}-set data")

    def __getitem__(self, index):
        item = self.lang_data[index]
        dataset = item[0]
        scan_id = item[1]
        sentence = item[2]

        # scene_pcds = self.scan_data[scan_id]['scene_pcds']

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) 
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            remained_obj_idx = [i for i in range(len(obj_pcds))]
            random.shuffle(remained_obj_idx)
            selected_obj_idxs = remained_obj_idx[:self.max_obj_len]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            assert len(obj_pcds) == self.max_obj_len

        if not self.aug_cfg:
            obj_fts, obj_locs, _, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True, rot_aug=self.rot_aug)
        else:
            obj_fts, obj_locs, _, obj_labels = self._obj_processing_aug(obj_pcds, obj_labels, is_need_bbox=True)

        data_dict = {'source': dataset,
                     'scan_id': scan_id,
                     'sentence': sentence,
                     # 'scene_pcds': scene_pcds,
                     'obj_fts': obj_fts,
                     'obj_locs': obj_locs,
                     'obj_labels': obj_labels} 
        return data_dict


@DATASET_REGISTRY.register()
class ScanNetSpatialRefer(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetSpatialRefer, self).__init__(cfg, split)

        self.pc_type = cfg.data.args.pc_type
        self.sem_type = cfg.data.args.sem_type
        self.max_obj_len = cfg.data.args.max_obj_len - 1
        self.num_points = cfg.data.args.num_points
        self.filter_lang = cfg.data.args.filter_lang
        self.rot_aug = cfg.data.args.rot_aug
        self.aug_cfg = getattr(cfg, 'data_aug', None)
        if self.aug_cfg:
            self.augmentor = DataAugmentor(self.aug_cfg, self.split)
        
        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)

        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        # assert self.anno_type in ['nr3d', 'sr3d']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val, ScanRefer and Referit3D Diff
        if self.split == 'test':
            self.split = 'val'

        split_scan_ids = self._load_split(cfg, self.split)

        print(f"Loading ScanNet SpatialRefer {split}-set language")
        split_cfg = cfg.data.get(self.__class__.__name__).get(split)
        self.lang_data, self.scan_ids = self._load_lang(split_cfg, split_scan_ids)
        print(f"Finish loading ScanNet SpatialRefer {split}-set language")

        print(f"Loading ScanNet SpatialRefer {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=self.split!='test')
        print(f"Finish loading ScanNet SpatialRefer {split}-set data")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count_multi'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])
            self.scan_data[scan_id]['label_count_hard'] = collections.Counter(
                                    [l for l in inst_labels])

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['utterance'].split(' '))

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
            gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_center = self.scan_data[scan_id]['obj_center']
            obj_box_size = self.scan_data[scan_id]['obj_box_size']
            obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
            obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
            for i, _ in enumerate(obj_center_pred):
                for j, _ in enumerate(obj_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                  obj_box_size[j]),
                                           construct_bbox_corners(obj_center_pred[i],
                                                                  obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break

        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_label = obj_labels[tgt_object_id]
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
            assert self.int2cat[tgt_object_label] == tgt_object_name, str(self.int2cat[tgt_object_label]) + '-' + tgt_object_name
        elif self.pc_type == 'pred':
            gt_pcd = self.scan_data[scan_id]["obj_pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        if not self.load_scene_pcds:
            if not self.aug_cfg:
                obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True, rot_aug=self.rot_aug)
            else:
                obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_aug(obj_pcds, obj_labels, is_need_bbox=True)
        else:
            assert self.aug_cfg
            if self.pc_type == 'pred':
                bg_pcds = self.scan_data[scan_id]['bg_pcds_pred']
            else:
                bg_pcds = self.scan_data[scan_id]['bg_pcds']
            obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds, bg_pcds, obj_labels, is_need_bbox=True)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long() if not self.load_scene_pcds else torch.zeros(len(obj_pcds) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long() if not self.load_scene_pcds else torch.zeros(len(obj_pcds) + 1).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count_multi'][self.label_converter.id_to_scannetid
                                                                  [tgt_object_label]] > 1
        is_hard = self.scan_data[scan_id]['label_count_hard'][tgt_object_label] > 2

        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard
        } if not self.load_scene_pcds else {
            "sentence": sentence, 
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "scene_pcds": scene_pcds, 
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N
            "obj_boxes": obj_boxes, # N, 6
            "data_idx": item_id, 
            "tgt_object_id_iou25": tgt_object_id_iou25, 
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            "is_multiple": is_multiple, 
            "is_view_dependent": is_view_dependent, 
            "is_hard": is_hard, 
            "obj_pcds_masks": obj_pcds_masks
        }

        return data_dict

    def _load_lang(self, cfg, split_scan_ids=None):
        lang_data = []
        sources = cfg.sources
        scan_ids = set()

        if sources:
            if 'referit3d' in sources:
                for anno_type in cfg.referit3d.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/{anno_type}.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                                scan_ids.add(item['scan_id'])
                                lang_data.append(item)

                if cfg.referit3d.sr3d_plus_aug:
                    anno_file = os.path.join(self.base_dir, 'annotations/refer/sr3d+.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                                scan_ids.add(item['scan_id'])
                                lang_data.append(item)
            if 'scanrefer' in sources:
                anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
                with jsonlines.open(anno_file, 'r') as _f:
                    for item in _f:
                        if item['scan_id'] in split_scan_ids:
                            scan_ids.add(item['scan_id'])
                            lang_data.append(item)
            if 'sgrefer' in sources:
                for anno_type in cfg.sgrefer.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_ref_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in split_scan_ids and item['instance_type'] not in ['wall', 'floor', 'ceiling']:
                            scan_ids.add(item['scan_id'])
                            lang_data.append(item)
            if 'sgcaption' in sources:
                for anno_type in cfg.sgcaption.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_caption_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in split_scan_ids and item['instance_type'] not in ['wall', 'floor', 'ceiling']:
                            scan_ids.add(item['scan_id'])
                            lang_data.append(item)

        return lang_data, scan_ids

@DATASET_REGISTRY.register()
class ScanNetPretrainObj(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetPretrainObj, self).__init__(cfg, split)
        self.pc_type = cfg.data.args.pc_type
        self.max_obj_len = cfg.data.args.max_obj_len
        self.num_points = cfg.data.args.num_points

        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)
        
        # TODO: only scanrefer needs test set
        if split != 'train':
            split = 'val'
        self.scannet_scan_ids = self._load_split(cfg, split)
        self.rot_aug = cfg.data.args.rot_aug
        self.aug_cfg = getattr(cfg, 'data_aug', None)
        if self.aug_cfg:
            self.augmentor = DataAugmentor(self.aug_cfg, self.split)

        print(f"Loading ScanNet {split}-set scans")
        self.scan_data = self._load_scannet(self.scannet_scan_ids, self.pc_type,
                                           load_inst_info = True)
        self.scan_ids = sorted(list(self.scan_data.keys()))
        print(f"Finish loading ScanNet {split}-set data")
    
    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        dataset = 'scannet'
        sentence = 'placeholder'

        # scene_pcds = self.scan_data[scan_id]['scene_pcds']

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) 
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            remained_obj_idx = [i for i in range(len(obj_pcds))]
            random.shuffle(remained_obj_idx)
            selected_obj_idxs = remained_obj_idx[:self.max_obj_len]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            assert len(obj_pcds) == self.max_obj_len

        if not self.load_scene_pcds:
            if not self.aug_cfg:
                obj_fts, obj_locs, _, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True, rot_aug=self.rot_aug)
            else:
                obj_fts, obj_locs, _, obj_labels = self._obj_processing_aug(obj_pcds, obj_labels, is_need_bbox=True)
        else:
            # if not self.aug_cfg:
            assert self.aug_cfg
            bg_pcds = self.scan_data[scan_id]['bg_pcds']
            obj_locs, _, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds, bg_pcds, obj_labels, is_need_bbox=True)

        if not self.load_scene_pcds:
            data_dict = {'source': dataset,
                        'scan_id': scan_id,
                        'sentence': sentence,
                        # 'scene_pcds': scene_pcds,
                        'obj_fts': obj_fts,
                        'obj_locs': obj_locs,
                        'obj_labels': obj_labels} 
        else:
            data_dict = {'source': dataset, 
                         'scan_id': scan_id, 
                         'sentence': sentence, 
                         'obj_locs': obj_locs, 
                         'obj_labels': obj_labels, 
                         'obj_pcds_masks': obj_pcds_masks, 
                         'scene_pcds': scene_pcds}
            
        return data_dict
