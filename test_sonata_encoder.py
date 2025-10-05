import torch
import spconv.pytorch as spconv
import numpy as np

# Assuming the model is in this file. If not, adjust the import path.
from modules.vision.sonata_encoder import Sonata3DSegLevelEncoder

def create_dummy_sparse_tensor(batch_size, num_points_per_batch, spatial_shape, in_channels):
    """Creates a dummy SparseConvTensor for testing."""
    features_list = []
    indices_list = []

    for i in range(batch_size):
        # Generate random voxel coordinates
        coords = np.random.randint(0, spatial_shape[0], size=(num_points_per_batch[i], len(spatial_shape)))
        # Ensure unique coordinates within the batch item
        coords = np.unique(coords, axis=0)
        
        # Add batch index
        batch_indices = np.full((coords.shape[0], 1), i)
        indices = np.hstack((batch_indices, coords))
        
        # Generate random features
        features = torch.randn(coords.shape[0], in_channels)
        
        indices_list.append(indices)
        features_list.append(features)

    # Concatenate all batches
    final_indices = torch.from_numpy(np.vstack(indices_list)).int()
    final_features = torch.cat(features_list, dim=0).float()

    # Create the sparse tensor
    sparse_tensor = spconv.SparseConvTensor(
        features=final_features,
        indices=final_indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )
    return sparse_tensor

def create_dummy_point2segment(num_points_per_batch, num_segments_per_batch):
    """Creates dummy point-to-segment mappings."""
    point2segment_list = []
    for n_points, n_segs in zip(num_points_per_batch, num_segments_per_batch):
        p2s = torch.randint(0, n_segs, (n_points,), dtype=torch.long)
        point2segment_list.append(p2s)
    max_seg = max(num_segments_per_batch)
    return point2segment_list, max_seg

def main():
    """Main function to run the test case."""
    # --- 1. Configuration ---
    batch_size = 2
    num_points_per_batch = [20000, 20000] # Number of points for each item in the batch
    num_segments_per_batch = [15, 20] # Number of segments for each item
    spatial_shape = [128, 128, 128] # Voxel grid size
    in_channels = 3  # Input feature channels (e.g., RGB)
    hidden_size = 256
    hlevels = [0, 1, 2, 3]

    # --- 2. Create Dummy Inputs ---
    print("Creating dummy data...")
    # Input sparse tensor (point cloud)
    # Note: The sonata backbone expects input features with 3 channels.
    sparse_input = create_dummy_sparse_tensor(batch_size, num_points_per_batch, spatial_shape, in_channels)
    
    # Point to segment mapping
    point2segment, max_seg = create_dummy_point2segment(num_points_per_batch, num_segments_per_batch)
    
    print(f"Batch size: {batch_size}")
    print(f"Total points: {sparse_input.features.shape[0]}")
    print(f"Max segments: {max_seg}")
    print("-" * 30)

    # --- 3. Instantiate the Model ---
    print("Instantiating model...")
    # Mock objects for configs
    class MockCfg:
        pass
    
    cfg = MockCfg()
    backbone_kwargs = {} # Assuming sonata.load doesn't need extra kwargs here

    model = Sonata3DSegLevelEncoder(
        cfg=cfg,
        backbone_kwargs=backbone_kwargs,
        hidden_size=hidden_size,
        hlevels=hlevels,
        freeze_backbone=False, # Set to True if you don't want to train the backbone
        dropout=0.1
    )
    model = model.cuda().eval() # Move to GPU and set to evaluation mode
    
    # Move sparse tensor and other inputs to GPU correctly
    sparse_input = sparse_input.to("cuda")
    point2segment = [p.cuda() for p in point2segment]
    
    print("Model instantiated successfully.")

    # --- 4. Run the Model ---
    print("Running model forward pass...")
    with torch.no_grad():
        output_feats = model(sparse_input, point2segment, max_seg)
    
    print("\n--- Test Passed! ---")
    print("Model forward pass completed without errors.")
    print(f"The model returned {len(output_feats)} feature tensors.")
    for i, feat in enumerate(output_feats):
        print(f"  - Output feature {i} shape: {feat.shape}")
    
    # Expected shape: [batch_size, max_seg, hidden_size]
    expected_shape = (batch_size, max_seg, hidden_size)
    for i, feat in enumerate(output_feats):
        # Note: The actual number of segments might be smaller than max_seg if some segments have no points.
        # So we check the first and last dimension.
        assert feat.shape[0] == expected_shape[0]
        assert feat.shape[2] == expected_shape[2]

if __name__ == "__main__":
    main()
