# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("../")  # noqa
import copy
import open3d as o3d
import sonata
import torch
import numpy as np
try:
    import flash_attn
    print(f"Flash attention version: {flash_attn.__version__}")
except ImportError:
    flash_attn = None


def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


if __name__ == "__main__":
    # set random seed
    # (random seed affect pca color, yet change random seed need manual adjustment kmeans)
    # (the pca prevent in paper is with another version of cuda and pytorch environment)
    sonata.utils.set_seed(53124)
    # Load model
    custom_config = dict(
          
           enc_mode=False
        )
    if flash_attn is not None:
        model = sonata.load("sonata_small", repo_id="facebook/sonata",custom_config=custom_config).cuda()
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()
    # Load default data transform pipeline
    transform = sonata.transform.default()
    # Load data
    point1 = sonata.data.load("sample1")
    point1.pop("segment200")
    segment = point1.pop("segment20")
    point1["segment"] = segment  # two kinds of segment exist in ScanNet, only use one
    print("shape of point1 segment:", point1["segment"].shape)
    print("shape of point1 coord:", point1["coord"].shape)
    original_coord = point1["coord"].copy()

    # import pdb; pdb.set_trace()

    point2 = copy.deepcopy(point1)
    # point2['color'] = np.zeros_like(point2['color'])
    

    point1 = transform(point1)
    point2 = transform(point2)
    point = sonata.data.collate_fn([point1, point2])

    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        # upcast point feature
        # Point is a structure contains all the information during forward
        print(point.keys())
        print(f"sparse_shape: {point.get('sparse_shape', 'Not found')}")
        
        # for _ in range(2):
        #     assert "pooling_parent" in point.keys()
        #     assert "pooling_inverse" in point.keys()
        #     parent = point.pop("pooling_parent")
        #     inverse = point.pop("pooling_inverse")
        #     point = parent
        #     print(f"After upcast {_+1}: {point['coord'].shape[0]} points, {point.feat.shape[-1]} channels")
        import pdb;pdb.set_trace()
        downsample_point = point
        scaled_multi_layer_feat = []
        inverse_map = []
        scaled_multi_layer_feat.append(downsample_point.feat)
        while "unpooling_parent" in downsample_point.keys():
            parent = downsample_point.pop("unpooling_parent")
            downsample_point = parent
            # scaled_multi_layer_feat.append(downsample_point.feat)
            inverse_map.append(downsample_point.pop("pooling_inverse"))
            for i in range(len(inverse_map)):
                downsample_point.feat= downsample_point.feat[inverse_map[len(inverse_map)-1 - i]]
            scaled_multi_layer_feat.append(downsample_point.feat)
            print("shape of current downsampled point feat:", downsample_point.feat.shape)
            print(f"After unpooling inverse: {downsample_point['coord'].shape[0]} points, {downsample_point.feat.shape[-1]} channels")
        
        # PCA
        pca_color = get_pca_color(point.feat, brightness=1.2, center=True)
        batched_coord = point.coord.clone()
        batched_coord[:, 0] += point.batch * 8.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(batched_coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    
    # Save point cloud to PLY file
    o3d.io.write_point_cloud("output_point_cloud.ply", pcd)
    print("Point cloud saved to output_point_cloud.ply")
