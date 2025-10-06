import torch
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch.hash import HashTable
import time

def align_features_to_target_coords_with_hashtable(features_upsampled, coors_upsampled, coors_target, spatial_shape):
    """
    使用 spconv 的 HashTable 将上采样后的特征与目标坐标对齐。

    参数:
        features_upsampled (torch.Tensor): 上采样后的特征张量 (N, C)。
        coors_upsampled (torch.Tensor): 上采样后的坐标张量 (N, 4)。
        coors_target (torch.Tensor): 目标坐标张量 (M, 4)。
        spatial_shape (list/tuple): 稀疏网格的空间尺寸。

    返回:
        torch.Tensor: 对齐后的特征张量 (M, C)。
    """
    
    max_z = spatial_shape[2]
    max_y = spatial_shape[1]
    
    def hash_coors(coors):
        coors = coors.long()
        return coors[:, 0] * (spatial_shape[0] * max_y * max_z) + \
               coors[:, 1] * (max_y * max_z) + \
               coors[:, 2] * max_z + \
               coors[:, 3]

    hashed_keys_upsampled = hash_coors(coors_upsampled)
    hashed_keys_target = hash_coors(coors_target)
    
    max_hash_size = int(hashed_keys_upsampled.numel() * 2)
    device = features_upsampled.device
    
    values_upsampled = torch.arange(
        hashed_keys_upsampled.numel(), dtype=torch.int32, device=device
    )
    
    hash_table = HashTable(
        device,
        torch.int64,
        torch.int32,
        max_size=max_hash_size
    )
    hash_table.insert(hashed_keys_upsampled, values_upsampled)
    
    # 获取哈希表内部的键和值
    keys, values, _ = hash_table.items()

    # 查询哈希表以获取哈希表内部值数组的索引
    query_table_indices, found = hash_table.query(hashed_keys_target)
    
    found_mask = found != -1
    
    # 使用正确的索引进行填充
    original_upsampled_indices = values[query_table_indices[found_mask]]
    
    aligned_features = torch.zeros(
        (coors_target.shape[0], features_upsampled.shape[1]),
        dtype=features_upsampled.dtype,
        device=device
    )
    
    aligned_features[found_mask] = features_upsampled[original_upsampled_indices]
    
    return aligned_features

# --- 测试主函数 ---
if __name__ == '__main__':
    # 1. 模拟 CUDA 设备上的输入数据
    # 这个设置与之前的例子完全相同。
    device = torch.device("cuda:0")
    batch_size = 2
    spatial_shape = [100, 100, 100]
    
    coors_target = torch.randint(0, 50, (2000, 4), dtype=torch.int32, device=device)
    features_input = torch.randn(coors_target.shape[0], 32, device=device)
    
    x_in = spconv.SparseConvTensor(features_input, coors_target, spatial_shape, batch_size)
    
    encoder = spconv.SparseSequential(
        spconv.SparseConv3d(32, 64, 3, 2, padding=1),
    ).to(device)
    
    decoder = spconv.SparseSequential(
        spconv.SparseConvTranspose3d(64, 32, 3, 2, padding=1),
    ).to(device)
    
    x_down = encoder(x_in)
    x_up = decoder(x_down)
    
    features_upsampled = x_up.features
    coors_upsampled = x_up.indices
    
    print(f"原始点云数量: {coors_target.shape[0]}")
    print(f"上采样后的点数量: {coors_upsampled.shape[0]}")
    print("下采样后的点云数量: ", x_down.indices.shape[0])
    
    # 2. 使用改进后的基于 HashTable 的函数进行对齐，并统计耗时
    start_time = time.time()
    aligned_features = align_features_to_target_coords_with_hashtable(
        features_upsampled, coors_upsampled, coors_target, spatial_shape
    )
    elapsed_time = time.time() - start_time
    print(f"特征对齐耗时: {elapsed_time:.6f} 秒")
    
    print(f"对齐后的特征数量: {aligned_features.shape[0]}")
    
    # 3. 验证
    # 我们随机选择一个目标坐标，检查它是否被正确对齐。
    rand_idx = np.random.randint(0, coors_target.shape[0])
    target_coord_to_check = coors_target[rand_idx]
    
    # 在上采样后的张量中找到对应的索引。
    upsampled_idx = torch.where(torch.all(coors_upsampled == target_coord_to_check, dim=1))
    
    if upsampled_idx[0].numel() > 0:
        found_upsampled_feature = features_upsampled[upsampled_idx[0].item()]
        aligned_feature = aligned_features[rand_idx]
        
        is_aligned = torch.allclose(found_upsampled_feature, aligned_feature)
        print(f"特征对齐成功？ {is_aligned}")
    else:
        print(f"警告: 在上采样后的张量中没有找到随机选择的目标点 {target_coord_to_check.cpu().numpy()}。")