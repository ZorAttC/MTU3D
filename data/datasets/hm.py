import os
import collections
import json
import random

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASET_REGISTRY
from ..data_utils import LabelConverter, build_rotate_mat
from ..data_utils import convert_pc_to_box, construct_bbox_corners, eval_ref_one_sample, is_explicitly_view_dependent
from .data_augmentor import DataAugmentor


class HMBase(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.base_dir = cfg.data.hm_base
        self.scannet_dir = cfg.data.scan_family_base

        self.int2cat = json.load(open(os.path.join(self.scannet_dir,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.scannet_dir,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

    def _load_split(self, split):
        split_file = os.path.join(self.base_dir, 'annotations/splits/'+ split + "_split.txt")
        scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
        scan_ids = sorted(scan_ids)

        return scan_ids

    def _load_hm_scan(self, scan_ids, filter_bkg=False):
        scans = {}
        for scan_id in tqdm(scan_ids):
            if not os.path.exists(os.path.join(self.base_dir, "scan_data",
                                               "pcd-align", f"{scan_id}.pth")):
                continue
            pcd_data = torch.load(os.path.join(self.base_dir, "scan_data",
                                               "pcd-align", f"{scan_id}.pth"))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
            points = points - np.mean(points, axis=0)
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # build obj_pcds
            inst_to_label = torch.load(os.path.join(self.base_dir, "scan_data",
                                                    "instance_id_to_label",
                                                    f"{scan_id}_inst_to_label.pth"))
            obj_pcds = []
            inst_ids = []
            inst_labels = []
            bg_indices = np.full((points.shape[0], ), 1, dtype=np.bool_)
            for inst_id in inst_to_label.keys():
                if inst_to_label[inst_id] in self.cat2int.keys():
                    mask = instance_labels == inst_id
                    if np.sum(mask) == 0:
                        continue
                    obj_pcds.append(pcds[mask])
                    inst_ids.append(inst_id)
                    inst_labels.append(self.cat2int[inst_to_label[inst_id]])
                    if inst_to_label[inst_id] not in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask] = False

            if filter_bkg:
                selected_obj_idxs = [i for i, obj_label in enumerate(inst_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
                if len(selected_obj_idxs) == 0:
                    continue
            scans[scan_id] = {}
            # scans[scan_id]['scene_pcds'] = pcds
            scans[scan_id]['obj_pcds'] = obj_pcds
            scans[scan_id]['inst_labels'] = inst_labels
            scans[scan_id]['inst_ids'] = inst_ids
            scans[scan_id]['bg_pcds'] = pcds[bg_indices]
        return scans

    def _load_lang(self, cfg, scan_ids, caption_source):
        json_data = []
        lang_data = []
        valid_scan_ids = []
        for anno_type in caption_source:
            if 'anno' == anno_type:
                anno_file = os.path.join(self.base_dir, 'annotations/hm3d.json')
                json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
            else:
                anno_file = os.path.join(self.base_dir, f'annotations/ssg_ref_{anno_type}.json')
                json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
        for item in json_data:
            if item['scan_id'] in scan_ids:
                lang_data.append(item)
                if item['scan_id'] not in valid_scan_ids:
                    valid_scan_ids.append(item['scan_id'])
        valid_scan_ids = sorted(valid_scan_ids)
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            valid_scan_ids = valid_scan_ids[:cfg.debug.debug_size]
            lang_data = [item for item in lang_data if item['scan_id'] in valid_scan_ids]

        return lang_data, valid_scan_ids

    def _obj_processing_post(self, obj_pcds, obj_labels, is_need_bbox=False, rot_aug=False):
        # rotate obj
        rot_matrix = build_rotate_mat(self.split, rot_aug)

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        return obj_fts, obj_locs, obj_boxes, obj_labels

    def _obj_processing_aug(self, obj_pcds, obj_labels, is_need_bbox=False):
        # calculate size
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds,
                                                'num_points': self.num_points}
                                               )
        obj_pcds = data_dict['obj_pcds']
        obj_sizes = data_dict['obj_sizes']
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for _i, obj_pcd in enumerate(obj_pcds):
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_sizes[_i]
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_sizes[_i]
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        return obj_fts, obj_locs, obj_boxes, obj_labels
    
    def _scene_processing_aug(self, obj_pcds, bg_pcds, obj_labels, is_need_bbox=False):
        # sample background points
        fg_points_num = len(obj_pcds) * self.num_points
        assert fg_points_num < self.max_pcd_num_points
        bg_points_num = min(self.max_pcd_num_points - fg_points_num, self.bg_points_num)
        assert len(bg_pcds) > 0
        assert bg_points_num > 0
        bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        bg_pcds = bg_pcds[bg_points_indices]

        # augment objects
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds, 
                                                'bg_pcds': bg_pcds, 
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        obj_sizes = data_dict['obj_sizes']
        bg_pcds = data_dict['bg_pcds']
        assert len(obj_pcds) * obj_pcds[0].shape[0] == fg_points_num

        # calculate location and generate scene_pcd
        obj_locs = []
        obj_boxes = []
        scene_pcds = []
        for _i, obj_pcd in enumerate(obj_pcds):
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_sizes[_i]
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_sizes[_i]
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # build scene pcd
            scene_pcds.extend(obj_pcd)

        # # sample background points
        # assert len(scene_pcds) < self.max_pcd_num_points
        # bg_points_num = min(self.max_pcd_num_points - len(scene_pcds), self.bg_points_num)
        # assert len(bg_pcds) > 0
        # assert bg_points_num > 0
        # bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        # scene_pcds.extend(bg_pcds[bg_points_indices])
        # assert len(scene_pcds) == self.max_pcd_num_points

        scene_pcds.extend(bg_pcds)

        # generate obj point indices masks
        obj_pcds_masks = []
        offset = 0
        for _j in range(len(obj_pcds)):
            mask = np.arange(self.num_points) + offset
            assert len(mask) == len(obj_pcds[_j])
            obj_pcds_masks.append(mask)
            offset += self.num_points

        # convert to torch
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)
        obj_pcds_masks = torch.from_numpy(np.array(obj_pcds_masks))
        scene_pcds = np.array(scene_pcds)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_pcds_masks.shape[0] == obj_locs.shape[0]

        return obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds


@DATASET_REGISTRY.register()
class HMPretrain(HMBase):
    def __init__(self, cfg, split):
        super(HMPretrain, self).__init__(cfg, split)
        self.pc_type = cfg.data.args.pc_type

        self.max_obj_len = cfg.data.args.max_obj_len
        self.num_points = cfg.data.args.num_points
        self.rot_aug = cfg.data.args.rot_aug
        self.aug_cfg = getattr(cfg, 'data_aug', None)
        if self.aug_cfg:
            self.augmentor = DataAugmentor(self.aug_cfg, self.split)

        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        sources = cfg.data.get(self.__class__.__name__).get(split).sources
        all_scan_ids = self._load_split(self.split)

        print(f"Loading HM3D {split}-set language")
        self.lang_data, self.scan_ids = self._load_lang(cfg, all_scan_ids, sources)
        print(f"Finish loading HM3D {split}-set language of size {self.__len__()}")

        print(f"Loading HM3D {split}-set scans")
        self.scan_data = self._load_hm_scan(self.scan_ids)
        print(f"Finish loading HM3D {split}-set scans")

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        dataset = 'hm3d'
        scan_id = item['scan_id']
        sentence = item['utterance']

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)]
                                # if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
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
                     'obj_fts': obj_fts,
                     'obj_locs': obj_locs,
                     'obj_labels': obj_labels} 

        return data_dict

@DATASET_REGISTRY.register()
class HMPretrainObj(HMBase):
    def __init__(self, cfg, split):
        super(HMPretrainObj, self).__init__(cfg, split)
        self.pc_type = cfg.data.args.pc_type

        self.max_obj_len = cfg.data.args.max_obj_len
        self.num_points = cfg.data.args.num_points
        self.rot_aug = cfg.data.args.rot_aug
        self.aug_cfg = getattr(cfg, 'data_aug', None)
        if self.aug_cfg:
            self.augmentor = DataAugmentor(self.aug_cfg, self.split)

        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)

        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        self.scan_ids = sorted(list(self._load_split(self.split)))
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            self.scan_ids = self.scan_ids[:cfg.debug.debug_size]

        print(f"Loading HM3D {split}-set scans")
        self.scan_data = self._load_hm_scan(self.scan_ids, filter_bkg=True)
        self.scan_ids = sorted(list(self.scan_data.keys()))
        print(f"Finish loading HM3D {split}-set scans of length {len(self.scan_ids)}")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """
        scan_id = self.scan_ids[index]
        dataset = 'hm3d'
        sentence = 'placeholder'

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)]
                                # if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
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
            assert self.aug_cfg
            bg_pcds = self.scan_data[scan_id]['bg_pcds']
            obj_locs, _, obj_labels, obj_pcds_masks, scene_pcds = self._scene_processing_aug(obj_pcds, bg_pcds, obj_labels, is_need_bbox=True)

        if not self.load_scene_pcds:
            data_dict = {'source': dataset,
                        'scan_id': scan_id,
                        'sentence': sentence,
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

@DATASET_REGISTRY.register()
class HMSpatialRefer(HMBase):
    def __init__(self, cfg, split):
        super(HMSpatialRefer, self).__init__(cfg, split)
        self.pc_type = cfg.data.args.pc_type

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

        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        sources = cfg.data.get(self.__class__.__name__).get(split).sources
        all_scan_ids = self._load_split(self.split)

        print(f"Loading HM3D {split}-set language")
        self.lang_data, self.scan_ids = self._load_lang(cfg, all_scan_ids, sources)
        print(f"Finish loading HM3D {split}-set language of size {self.__len__()}")

        print(f"Loading HM3D {split}-set scans")
        self.scan_data = self._load_hm_scan(self.scan_ids)
        print(f"Finish loading HM3D {split}-set scans")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [l for l in inst_labels])

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_instance = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['utterance'].split(' '))

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_ids = self.scan_data[scan_id]['inst_ids'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            obj_ids = self.scan_data[scan_id]['inst_ids_pred'] # N

        assert tgt_object_instance in obj_ids, str(tgt_object_instance) + ' not in '+ '-' + scan_id
        tgt_object_id = obj_ids.index(tgt_object_instance)
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
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
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
        elif self.pc_type == 'pred':
            gt_pcd = self.scan_data[scan_id]["pcds"][tgt_object_id]
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
        is_multiple = self.scan_data[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

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
