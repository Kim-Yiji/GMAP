"""
Test threat score calculation with actual SDD dataset.

This script loads SDD dataset and computes threat scores for real trajectory data.
"""

import os
import torch
import numpy as np
import pandas as pd
from utils.threat_score import compute_threat_score_batch, get_obstacle_size
from collections import defaultdict


def load_sdd_annotations(annotations_file):
    """
    Load SDD annotations from CSV file.
    
    Args:
        annotations_file: Path to annotations.csv file
    
    Returns:
        trajectories: Dict mapping track_id to list of (frame, x, y) tuples
        labels: Dict mapping track_id to label (object type)
    """
    df = pd.read_csv(annotations_file)
    
    # Extract center coordinates from bounding boxes
    df['x'] = (df['xmin'] + df['xmax']) / 2.0
    df['y'] = (df['ymin'] + df['ymax']) / 2.0
    
    # Group by track_id
    trajectories = defaultdict(list)
    labels = {}
    
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        
        # Filter out lost/occluded frames
        track_data = track_data[(track_data['lost'] == 0) & (track_data['occluded'] == 0)]
        
        if len(track_data) < 2:  # Need at least 2 frames for velocity
            continue
        
        # Extract trajectory
        traj = [(row['frame'], row['x'], row['y']) for _, row in track_data.iterrows()]
        trajectories[track_id] = traj
        
        # Get label (object type)
        label = track_data['label'].iloc[0]
        labels[track_id] = label
    
    return trajectories, labels


def trajectories_to_tensors(trajectories, labels, min_seq_len=8, max_seq_len=20):
    """
    Convert trajectories to tensors compatible with threat score calculation.
    
    Args:
        trajectories: Dict mapping track_id to list of (frame, x, y) tuples
        labels: Dict mapping track_id to label
        min_seq_len: Minimum sequence length to include
        max_seq_len: Maximum sequence length to use
    
    Returns:
        obs_traj: Absolute positions, shape (num_peds, 2, seq_len)
        obs_traj_rel: Relative velocities, shape (num_peds, 2, seq_len)
        obstacle_sizes: Size values, shape (num_peds,)
        track_ids: List of track_ids in order
    """
    # Filter trajectories by length
    valid_tracks = {tid: traj for tid, traj in trajectories.items() 
                    if len(traj) >= min_seq_len}
    
    if len(valid_tracks) == 0:
        return None, None, None, None
    
    # Use the first max_seq_len frames
    seq_len = min(max_seq_len, min(len(traj) for traj in valid_tracks.values()))
    
    num_peds = len(valid_tracks)
    track_ids = list(valid_tracks.keys())
    
    # Initialize tensors
    obs_traj = torch.zeros((num_peds, 2, seq_len), dtype=torch.float32)
    obs_traj_rel = torch.zeros((num_peds, 2, seq_len), dtype=torch.float32)
    obstacle_sizes = torch.zeros(num_peds, dtype=torch.float32)
    
    # Fill tensors
    for i, track_id in enumerate(track_ids):
        traj = valid_tracks[track_id][:seq_len]
        
        # Extract positions
        positions = np.array([(x, y) for _, x, y in traj])
        
        # Store absolute positions
        obs_traj[i, 0, :] = torch.tensor(positions[:, 0], dtype=torch.float32)
        obs_traj[i, 1, :] = torch.tensor(positions[:, 1], dtype=torch.float32)
        
        # Compute relative velocities (frame-to-frame differences)
        velocities = np.diff(positions, axis=0)
        # Pad first frame with zero velocity
        velocities = np.vstack([np.zeros((1, 2)), velocities])
        
        obs_traj_rel[i, 0, :] = torch.tensor(velocities[:, 0], dtype=torch.float32)
        obs_traj_rel[i, 1, :] = torch.tensor(velocities[:, 1], dtype=torch.float32)
        
        # Get obstacle size from label
        label = labels[track_id]
        obstacle_sizes[i] = get_obstacle_size(label, default_size=0.0)
    
    return obs_traj, obs_traj_rel, obstacle_sizes, track_ids


def test_sdd_dataset(scene_name='bookstore', video_id='video0'):
    """
    Test threat score calculation with SDD dataset.
    
    Args:
        scene_name: Name of the scene (e.g., 'bookstore', 'coupa', 'deathCircle')
        video_id: Video ID (e.g., 'video0', 'video1')
    """
    print("=" * 80)
    print(f"Testing Threat Score Calculation with SDD Dataset")
    print(f"Scene: {scene_name}, Video: {video_id}")
    print("=" * 80)
    
    # Path to annotations file
    annotations_file = f'/raid/guest/OATMeal_Queens/SDD_datasets/SDD_raw/{scene_name}/{video_id}/annotations.csv'
    
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file not found: {annotations_file}")
        return
    
    print(f"\nLoading annotations from: {annotations_file}")
    
    # Load trajectories
    trajectories, labels = load_sdd_annotations(annotations_file)
    
    print(f"\nLoaded {len(trajectories)} trajectories")
    print(f"Object types: {set(labels.values())}")
    
    # Convert to tensors
    obs_traj, obs_traj_rel, obstacle_sizes, track_ids = trajectories_to_tensors(
        trajectories, labels, min_seq_len=8, max_seq_len=20
    )
    
    if obs_traj is None:
        print("Error: No valid trajectories found")
        return
    
    num_peds = obs_traj.shape[0]
    seq_len = obs_traj.shape[2]
    
    print(f"\nTrajectory data:")
    print(f"  Number of objects: {num_peds}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Position range: x=[{obs_traj[:, 0, :].min():.2f}, {obs_traj[:, 0, :].max():.2f}], "
          f"y=[{obs_traj[:, 1, :].min():.2f}, {obs_traj[:, 1, :].max():.2f}]")
    print(f"  Velocity range: x=[{obs_traj_rel[:, 0, :].min():.2f}, {obs_traj_rel[:, 0, :].max():.2f}], "
          f"y=[{obs_traj_rel[:, 1, :].min():.2f}, {obs_traj_rel[:, 1, :].max():.2f}]")
    
    print(f"\nObstacle sizes:")
    for i, track_id in enumerate(track_ids):
        label = labels[track_id]
        size = obstacle_sizes[i].item()
        print(f"  Track {track_id} ({label}): size = {size:.2f}")
    
    # Get object labels for pedestrian mask
    object_labels = [labels[track_id] for track_id in track_ids]
    
    # Compute threat scores (only for pedestrians perceiving threats)
    print(f"\nComputing threat scores...")
    print(f"  Note: Only computing threats perceived by pedestrians (not by vehicles/obstacles)")
    threat_score, z_ij = compute_threat_score_batch(
        obs_traj,
        obs_traj_rel,
        obstacle_sizes=obstacle_sizes,
        weights=None,  # Use distance-weighted [0.5, 0.25, 0.15, 0.1] (distance has largest impact)
        tau=0.15,  # Temperature parameter
        beta=0.5,  # Midpoint parameter
        object_labels=object_labels  # Used to identify pedestrians
    )
    
    # Identify pedestrians for statistics
    pedestrian_types = {'pedestrian', 'person'}
    pedestrian_mask = [labels[track_id].lower() in pedestrian_types for track_id in track_ids]
    pedestrian_mask_tensor = torch.tensor(pedestrian_mask, dtype=torch.bool)
    pedestrian_mask_expanded = pedestrian_mask_tensor.unsqueeze(1).unsqueeze(2).expand(-1, num_peds, seq_len)
    threat_pedestrians_only = threat_score[pedestrian_mask_expanded]
    
    print(f"\nThreat Score Results:")
    print(f"  Shape: {threat_score.shape}  # (num_peds, num_peds, seq_len)")
    print(f"  Range: [{threat_score.min():.4f}, {threat_score.max():.4f}]")
    print(f"  Mean (pedestrians only): {threat_pedestrians_only.mean():.4f}")
    print(f"  Std (pedestrians only): {threat_pedestrians_only.std():.4f}")
    
    print(f"\nFeature Vector z_ij Components:")
    print(f"  Distance d_ij: range [{z_ij[:, :, 0, :].min():.4f}, {z_ij[:, :, 0, :].max():.4f}], "
          f"mean={z_ij[:, :, 0, :].mean():.4f}")
    print(f"  Approach velocity v+_ij: range [{z_ij[:, :, 1, :].min():.4f}, {z_ij[:, :, 1, :].max():.4f}], "
          f"mean={z_ij[:, :, 1, :].mean():.4f}")
    print(f"  Obstacle size size_j: range [{z_ij[:, :, 2, :].min():.4f}, {z_ij[:, :, 2, :].max():.4f}], "
          f"mean={z_ij[:, :, 2, :].mean():.4f}")
    print(f"  TTC TTC_ij: range [{z_ij[:, :, 3, :].min():.4f}, {z_ij[:, :, 3, :].max():.4f}], "
          f"mean={z_ij[:, :, 3, :].mean():.4f}")
    
    # Show threat scores for first frame
    print(f"\nThreat Scores at Frame 0:")
    print(f"  Matrix (row=object i, col=obstacle j):")
    threat_matrix = threat_score[:, :, 0].numpy()
    print(f"  Shape: {threat_matrix.shape}")
    
    # Find top threat pairs
    eye_mask = np.eye(num_peds, dtype=bool)
    threat_matrix_no_self = threat_matrix.copy()
    threat_matrix_no_self[eye_mask] = 0.0
    
    if threat_matrix_no_self.max() > 0:
        max_idx = np.unravel_index(np.argmax(threat_matrix_no_self), threat_matrix_no_self.shape)
        i, j = max_idx
        print(f"\n  Highest threat pair:")
        print(f"    Object {i} (track {track_ids[i]}, {labels[track_ids[i]]}) -> "
              f"Obstacle {j} (track {track_ids[j]}, {labels[track_ids[j]]})")
        print(f"    Threat score: {threat_matrix[i, j]:.4f}")
        
        # Show z_ij components for this pair
        z_ij_val = z_ij[i, j, :, 0]
        print(f"    z_{i}{j} = [d={z_ij_val[0]:.4f}, v+={z_ij_val[1]:.4f}, "
              f"size={z_ij_val[2]:.4f}, TTC={z_ij_val[3]:.4f}]")
    
    # Show threat scores over time for a specific pair
    if num_peds >= 2:
        i, j = 0, 1
        print(f"\n  Threat score over time for pair (i={i}, j={j}):")
        threat_over_time = threat_score[i, j, :].numpy()
        print(f"    Frame-by-frame: {threat_over_time}")
        print(f"    Mean: {threat_over_time.mean():.4f}, Max: {threat_over_time.max():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Threat score calculation completed successfully!")
    print("=" * 80)
    
    return threat_score, z_ij, track_ids, labels


if __name__ == "__main__":
    import sys
    
    # Test with different scenes
    scenes_to_test = [
        ('bookstore', 'video0'),
        ('coupa', 'video0'),
        ('deathCircle', 'video0'),
    ]
    
    if len(sys.argv) > 1:
        scene_name = sys.argv[1]
        video_id = sys.argv[2] if len(sys.argv) > 2 else 'video0'
        test_sdd_dataset(scene_name, video_id)
    else:
        # Test all scenes
        for scene_name, video_id in scenes_to_test:
            try:
                test_sdd_dataset(scene_name, video_id)
                print("\n")
            except Exception as e:
                print(f"Error testing {scene_name}/{video_id}: {e}")
                import traceback
                traceback.print_exc()
                print("\n")

