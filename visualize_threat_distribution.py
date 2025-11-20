"""
Visualize threat score distribution for a single SDD dataset scene.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.threat_score import compute_threat_score_batch, get_obstacle_size


def load_sdd_annotations(annotations_file):
    """Load SDD annotations from CSV file."""
    df = pd.read_csv(annotations_file)
    df['x'] = (df['xmin'] + df['xmax']) / 2.0
    df['y'] = (df['ymin'] + df['ymax']) / 2.0
    
    trajectories = defaultdict(list)
    labels = {}
    
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        track_data = track_data[(track_data['lost'] == 0) & (track_data['occluded'] == 0)]
        
        if len(track_data) < 2:
            continue
        
        traj = [(row['frame'], row['x'], row['y']) for _, row in track_data.iterrows()]
        trajectories[track_id] = traj
        labels[track_id] = track_data['label'].iloc[0]
    
    return trajectories, labels


def trajectories_to_tensors(trajectories, labels, min_seq_len=8, max_seq_len=20):
    """Convert trajectories to tensors."""
    valid_tracks = {tid: traj for tid, traj in trajectories.items() if len(traj) >= min_seq_len}
    
    if len(valid_tracks) == 0:
        return None, None, None, None
    
    seq_len = min(max_seq_len, min(len(traj) for traj in valid_tracks.values()))
    num_peds = len(valid_tracks)
    track_ids = list(valid_tracks.keys())
    
    obs_traj = torch.zeros((num_peds, 2, seq_len), dtype=torch.float32)
    obs_traj_rel = torch.zeros((num_peds, 2, seq_len), dtype=torch.float32)
    obstacle_sizes = torch.zeros(num_peds, dtype=torch.float32)
    
    for i, track_id in enumerate(track_ids):
        traj = valid_tracks[track_id][:seq_len]
        positions = np.array([(x, y) for _, x, y in traj])
        obs_traj[i, 0, :] = torch.tensor(positions[:, 0], dtype=torch.float32)
        obs_traj[i, 1, :] = torch.tensor(positions[:, 1], dtype=torch.float32)
        
        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([np.zeros((1, 2)), velocities])
        obs_traj_rel[i, 0, :] = torch.tensor(velocities[:, 0], dtype=torch.float32)
        obs_traj_rel[i, 1, :] = torch.tensor(velocities[:, 1], dtype=torch.float32)
        
        obstacle_sizes[i] = get_obstacle_size(labels[track_id], default_size=0.0)
    
    return obs_traj, obs_traj_rel, obstacle_sizes, track_ids


def visualize_threat_distribution(scene_name='bookstore', video_id='video0', save_path=None):
    """Visualize threat score distribution for a dataset."""
    print(f"Loading {scene_name}/{video_id}...")
    
    annotations_file = f'/raid/guest/OATMeal_Queens/SDD_datasets/SDD_raw/{scene_name}/{video_id}/annotations.csv'
    
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file not found: {annotations_file}")
        return
    
    # Load data
    trajectories, labels = load_sdd_annotations(annotations_file)
    obs_traj, obs_traj_rel, obstacle_sizes, track_ids = trajectories_to_tensors(
        trajectories, labels, min_seq_len=8, max_seq_len=20
    )
    
    if obs_traj is None:
        print("Error: No valid trajectories found")
        return
    
    # Get object labels
    object_labels = [labels[track_id] for track_id in track_ids]
    
    # Compute threat scores
    threat_score, z_ij = compute_threat_score_batch(
        obs_traj, obs_traj_rel,
        obstacle_sizes=obstacle_sizes,
        weights=None,
        tau=0.15,
        beta=0.5,
        object_labels=object_labels
    )
    
    num_peds = threat_score.shape[0]
    seq_len = threat_score.shape[2]
    
    # Filter to pedestrians only
    pedestrian_types = {'pedestrian', 'person'}
    pedestrian_mask = [labels[track_id].lower() in pedestrian_types for track_id in track_ids]
    pedestrian_mask_tensor = torch.tensor(pedestrian_mask, dtype=torch.bool)
    pedestrian_mask_expanded = pedestrian_mask_tensor.unsqueeze(1).unsqueeze(2).expand(-1, num_peds, seq_len)
    threat_pedestrians_only = threat_score[pedestrian_mask_expanded].numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram
    axes[0].hist(threat_pedestrians_only, bins=50, range=(0, 1), 
                 alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    axes[0].axvline(threat_pedestrians_only.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {threat_pedestrians_only.mean():.3f}')
    axes[0].set_xlabel('Threat Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Threat Score Distribution\n({scene_name}/{video_id})', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. CDF
    sorted_scores = np.sort(threat_pedestrians_only)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cumulative, linewidth=2, color='purple')
    axes[1].axvline(threat_pedestrians_only.mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Threat Score', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import sys
    
    scene_name = sys.argv[1] if len(sys.argv) > 1 else 'bookstore'
    video_id = sys.argv[2] if len(sys.argv) > 2 else 'video0'
    save_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    visualize_threat_distribution(scene_name, video_id, save_path)

