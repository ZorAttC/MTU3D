import os
import collections
import json
import pickle
import random
import jsonlines
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset

from common.misc import rgetattr
from data.datasets.scannet_base import ScanNetBase
from ..data_utils import convert_pc_to_box, LabelConverter, build_rotate_mat, load_matrix_from_txt, \
                        construct_bbox_corners, eval_ref_one_sample  
from copy import deepcopy


RSCAN_DATA = {}

class RScanBase(ScanNetBase):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.base_dir = cfg.data.rscan_base
        self.load_scan_options = self.cfg.data.get('load_scan_options', {})
        assert self.split in ['train', 'val', 'test']

        self.int2cat = json.load(open(os.path.join(cfg.data.scan_family_base,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(cfg.data.scan_family_base,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))
            
    def _load_one_scan(self, scan_id):
        options = self.load_scan_options
        one_scan = {}

        if options.get('load_pc_info', True):
            # load pcd data
            pcd_data = torch.load(os.path.join(self.base_dir, "3RScan-ours-align/", scan_id, 'pcd-align.pth'))
            inst_to_sem_name = torch.load(os.path.join(self.base_dir, "3RScan-ours-align/", scan_id, 'inst_to_label.pth'))
            points, colors, inst_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            one_scan['pcds'] = pcds
            one_scan['inst_to_sem_name'] = inst_to_sem_name
            # build semantic labels in scannet raw format
            sem_labels = np.zeros(points.shape[0]) - 1
            for inst_id in inst_to_sem_name.keys():
                if inst_to_sem_name[inst_id] in self.cat2int.keys():
                    mask = inst_labels == inst_id
                    if np.sum(mask) == 0:
                        continue
                    sem_labels[mask] = int(self.label_converter.raw_name_to_scannet_raw_id[inst_to_sem_name[inst_id]])
            one_scan['sem_labels'] = sem_labels        
            # build inst label mapper to map inst label to 0...max
            inst_label_mapper = {}
            max_inst_id = 0
            for inst_id in np.unique(inst_labels):
                if inst_id in inst_to_sem_name.keys():
                    inst_label_mapper[inst_id] = max_inst_id
                    max_inst_id += 1
                else:
                    inst_label_mapper[inst_id] = -1 # -1 for no object
            one_scan['inst_label_mapper'] = inst_label_mapper
            inst_labels = np.vectorize(lambda x: inst_label_mapper.get(x, x))(inst_labels)
            one_scan['inst_labels'] = inst_labels
            # convert to gt object
            if options.get('load_inst_info', True):

                def get_loc(obj_pcd):
                    center = obj_pcd[:, :3].mean(0)
                    size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                    return list(center) + list(size)
                
                def get_box(obj_pcd):
                    center, size = convert_pc_to_box(obj_pcd)
                    return list(center) + list(size)
                
                # gt masks
                obj_masks = inst_labels[None, :] == np.arange(inst_labels.max() + 1)[:, None] if inst_labels is not None else [] #  test
                obj_boxes = np.array([get_box(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                obj_locs = np.array([get_loc(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                bg_obj_ids = [inst_label_mapper[inst_id] for inst_id in inst_to_sem_name.keys() if inst_to_sem_name[inst_id] in ['wall', 'floor', 'ceiling']]
                bg_mask = [] if obj_masks == [] else obj_masks[bg_obj_ids].sum(0).astype(bool) # ScanQA test
                one_scan['obj_masks'] = [np.nonzero(mask)[0] for mask in obj_masks]
                one_scan['bg_mask'] = bg_mask
                one_scan['obj_loc'] = obj_locs
                one_scan['obj_box'] = obj_boxes
                 
                # TODO: Add predicted mask 
                # obj_mask_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".mask" + ".npz")
                # if os.path.exists(obj_mask_path):
                #     obj_label_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".label" + ".npy")

                #     topk = 50
                #     obj_masks = np.array(sparse.load_npz(obj_mask_path).todense(), dtype=bool)[:topk] # TODO: only keep the top-50 bbox for now
                #     inst_labels = np.load(obj_label_path)[:topk]
                #     valid = obj_masks.any(axis=1)
                #     obj_masks = obj_masks[valid]
                #     inst_labels = inst_labels[valid]
                #     obj_boxes = np.array([get_box(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                #     obj_locs = np.array([get_loc(pcds[mask]) for mask in obj_masks], dtype=np.float32)
                #     bg_mask = ~(obj_masks.sum(axis=0) > 0)
                #     one_scan['obj_masks_pred'] = [np.nonzero(mask)[0] for mask in obj_masks]
                #     one_scan['bg_mask_pred'] = bg_mask
                #     one_scan['obj_loc_pred'] = obj_locs
                #     one_scan['obj_box_pred'] = obj_boxes
                #     one_scan['inst_labels_pred'] = inst_labels
                #     self.match_gt_to_pred(one_scan)
                
        # if options.get('load_voxel_feats', False):
        #     one_scan['voxel_feats'] = self._load_voxel_feats(scan_id)
        #     one_scan['voxel_feats_pred'] = self._load_voxel_feats(scan_id, gt=False)

        # if options.get('load_mv_feats', False):
        #     one_scan['mv_feats'] = self._load_mv_feats(scan_id)
        #     one_scan['mv_feats_pred'] = self._load_mv_feats(scan_id, gt=False)
        
        # load segment for mask3d
        if options.get('load_segment_info', False):
            segment_id = np.load(os.path.join(self.base_dir, 'segment_id', f'{scan_id}.npy'))
            one_scan['segment_id'] = segment_id
            assert segment_id.shape[0] == one_scan['pcds'].shape[0]
                    
        # # load offline feature 
        # if options.get('load_offline_segment_voxel', False):
        #     one_scan['offline_segment_voxel'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_voxel_feature", f'{scan_id}.pth'), map_location='cpu')
            
        if options.get('load_offline_segment_image', False):
            one_scan['offline_segment_image'] = torch.load(os.path.join(self.base_dir, "3RScan-features", "image_seg_feat", f'{scan_id}.pth'), map_location='cpu')
            
        if options.get('load_offline_segment_point', False):
            one_scan['offline_segment_point'] = torch.load(os.path.join(self.base_dir, "3RScan-features", "point_seg_feat", f'{scan_id}.pth'), map_location='cpu')

        return (scan_id, one_scan)

    def _load_scannet(self, scan_ids):
        process_num = self.load_scan_options.get('process_num', 0)
        unloaded_scan_ids = [scan_id for scan_id in scan_ids if scan_id not in RSCAN_DATA]
        print(f"Loading scans: {len(unloaded_scan_ids)} / {len(scan_ids)}")
        scans = {}
        if process_num > 1:
            from joblib import Parallel, delayed
            res_all = Parallel(n_jobs=process_num)(
                delayed(self._load_one_scan)(scan_id) for scan_id in tqdm(unloaded_scan_ids))
            for scan_id, one_scan in tqdm(res_all):
                scans[scan_id] = one_scan
        else:
            for scan_id in tqdm(unloaded_scan_ids):
                _, one_scan = self._load_one_scan(scan_id)
                scans[scan_id] = one_scan

        RSCAN_DATA.update(scans)
        scans = {scan_id: RSCAN_DATA[scan_id] for scan_id in scan_ids}
        return scans
    
    def _load_voxel_feats(self, scan_id, gt = True):
        mask_type = '_GT' if gt else ''
        feat_pth = os.path.join(self.base_dir, 'mask3d_obj_centric_feats' + mask_type, scan_id + '.pth')
        if not os.path.exists(feat_pth):
            return None
        feat_dict = torch.load(feat_pth)
        feat_dim = next(iter(feat_dict.values())).shape[0]
        n_obj = max(feat_dict.keys()) + 1 # the last one is for missing objects.
        feat = torch.zeros((n_obj, feat_dim), dtype=torch.float32)
        for k, v in feat_dict.items():
            feat[k] = v

        return feat
    
    def _load_mv_feats(self, scan_id, gt = True):
        mask_type = 'gt' if gt else 'pred'
        feat_pth = os.path.join(self.base_dir, f'openseg_obj_feats_{mask_type}_mask', scan_id + '.pt')        
        if not os.path.exists(feat_pth):
            return None
        feat = torch.load(feat_pth)
        feat = torch.cat((feat, torch.zeros_like(feat[:1])), dim=0) # the last one is for missing objects.
        return feat

    def _load_split(self, cfg, split, use_multi_process = False):
        split_file = os.path.join(self.base_dir, "3RScan-meta", "3RScan.json") 
        rscan_scan_ids = []
        ignore_scan_ids = ['4e858ca1-fd93-2cb4-84b6-490997979830', 'd7d40d62-7a5d-2b36-955e-86a394caeabb', '4a9a43d8-7736-2874-86fc-098deb94c868']
        
        # exclude scenes with too many segments during training
        if split == 'train':
            exlcude_ids_file = os.path.join(self.base_dir, "3RScan-meta", "exclude_ids.json")
            with open(exlcude_ids_file, 'r') as f:
                exclude_ids = json.load(f)['3RScan']
                ignore_scan_ids += exlcude_ids_file
            
        with open(split_file, 'r') as f:
            rscan_json = json.load(f)
            for scene in rscan_json:
                if split in scene['type']:
                    rscan_scan_ids.append(scene['reference'])
                    for sub_scene in scene['scans']:
                        rscan_scan_ids.append(sub_scene['reference'])
        
        rscan_scan_ids = list(set(rscan_scan_ids) - set(ignore_scan_ids))
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            rscan_scan_ids = list(rscan_scan_ids)[:cfg.debug.debug_size]
            
        return rscan_scan_ids

    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        self.pc_type = dataset_cfg.get('pc_type', 'gt')
        self.sem_type = dataset_cfg.get('sem_type', '607')
        self.max_obj_len = dataset_cfg.get('max_obj_len', 80)
        self.num_points = dataset_cfg.get('num_points', 1024)
        self.filter_lang = dataset_cfg.get('filter_lang', False)
        self.rot_aug = dataset_cfg.get('rot_aug', True)
        self.train_duplicate = dataset_cfg.get('train_duplicate', 1)
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']

    def init_scan_data(self):
        self.scan_data = self._load_scannet(self.scan_ids)

        # build unique multiple look up
        if self.load_scan_options.get('load_inst_info', True):
            for scan_id in self.scan_ids:
                inst_to_sem_name = self.scan_data[scan_id]['inst_to_sem_name']
                self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[self.cat2int[v]] for v in inst_to_sem_name.values()])
        