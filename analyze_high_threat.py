"""
Analyze high threat score objects in SDD dataset.
"""

import torch
import numpy as np
import pandas as pd
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
    valid_tracks = {tid: traj for tid, traj in trajectories.items() 
                    if len(traj) >= min_seq_len}
    
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
        
        label = labels[track_id]
        obstacle_sizes[i] = get_obstacle_size(label, default_size=0.0)
    
    return obs_traj, obs_traj_rel, obstacle_sizes, track_ids


def analyze_high_threat(scene_name='coupa', video_id='video0', threshold=0.4):
    """Analyze objects with high threat scores."""
    print("=" * 80)
    print(f"Analyzing High Threat Scores (threshold >= {threshold})")
    print(f"Scene: {scene_name}, Video: {video_id}")
    print("=" * 80)
    
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
    
    # Get object labels for pedestrian identification
    object_labels = [labels[track_id] for track_id in track_ids]
    
    # Compute threat scores (only pedestrians perceive threats)
    threat_score, z_ij = compute_threat_score_batch(
        obs_traj, obs_traj_rel,
        obstacle_sizes=obstacle_sizes,
        weights=None,
        tau=0.15,
        beta=0.5,
        object_labels=object_labels  # Pass labels to identify pedestrians
    )
    
    num_peds = threat_score.shape[0]
    seq_len = threat_score.shape[2]
    
    # Identify pedestrians (only these perceive threats)
    # Only 'pedestrian' and 'person' are considered pedestrians
    # Biker, Skater, Car, Bus, etc. are NOT pedestrians
    pedestrian_types = {'pedestrian', 'person'}
    pedestrian_mask = [labels[track_id].lower() in pedestrian_types for track_id in track_ids]
    pedestrian_indices = [i for i, is_ped in enumerate(pedestrian_mask) if is_ped]
    
    # Calculate statistics for pedestrians only
    pedestrian_mask_tensor = torch.tensor(pedestrian_mask, dtype=torch.bool)
    pedestrian_mask_expanded = pedestrian_mask_tensor.unsqueeze(1).unsqueeze(2).expand(-1, num_peds, seq_len)
    threat_pedestrians_only = threat_score[pedestrian_mask_expanded]
    
    print(f"\nThreat Score Statistics:")
    print(f"  Shape: {threat_score.shape}")
    print(f"  Range: [{threat_score.min():.4f}, {threat_score.max():.4f}]")
    print(f"  Mean (pedestrians only): {threat_pedestrians_only.mean():.4f}")
    print(f"  Std (pedestrians only): {threat_pedestrians_only.std():.4f}")
    
    print(f"\nObject Classification:")
    print(f"  Pedestrians (perceive threats): {len(pedestrian_indices)}")
    print(f"  Non-pedestrians (do not perceive threats): {num_peds - len(pedestrian_indices)}")
    
    # Find high threat pairs (ONLY for pedestrians)
    eye_mask = torch.eye(num_peds, dtype=torch.bool)
    eye_mask = eye_mask.unsqueeze(2).expand(-1, -1, seq_len)
    threat_no_self = threat_score.clone()
    threat_no_self[eye_mask] = 0.0
    
    # Only consider threats perceived by pedestrians
    pedestrian_mask_tensor = torch.tensor(pedestrian_mask, dtype=torch.bool)
    pedestrian_mask_expanded = pedestrian_mask_tensor.unsqueeze(1).unsqueeze(2).expand(-1, num_peds, seq_len)
    threat_pedestrians_only = threat_no_self.clone()
    threat_pedestrians_only[~pedestrian_mask_expanded] = 0.0
    
    high_threat_mask = threat_pedestrians_only >= threshold
    num_high_threat = high_threat_mask.sum().item()
    
    # Calculate percentage based on pedestrian pairs only
    num_pedestrian_pairs = len(pedestrian_indices) * num_peds * seq_len - len(pedestrian_indices) * seq_len
    
    print(f"\nHigh Threat Analysis (threshold >= {threshold}):")
    print(f"  Total high threat pairs (pedestrians only): {num_high_threat}")
    print(f"  Percentage (of pedestrian pairs): {100 * num_high_threat / num_pedestrian_pairs:.2f}%")
    
    # Analyze by object type (ONLY for pedestrians perceiving threats)
    high_threat_by_type = defaultdict(list)
    high_threat_details = []
    
    for i in range(num_peds):
        # Skip if i is not a pedestrian (non-pedestrians don't perceive threats)
        if not pedestrian_mask[i]:
            continue
            
        for j in range(num_peds):
            if i == j:
                continue
            for t in range(seq_len):
                if threat_score[i, j, t] >= threshold:
                    obj_i_type = labels[track_ids[i]]
                    obj_j_type = labels[track_ids[j]]
                    score = threat_score[i, j, t].item()
                    
                    high_threat_by_type[(obj_i_type, obj_j_type)].append(score)
                    high_threat_details.append({
                        'pedestrian_i': track_ids[i],
                        'pedestrian_type_i': obj_i_type,
                        'obstacle_j': track_ids[j],
                        'obstacle_type_j': obj_j_type,
                        'frame': t,
                        'threat_score': score,
                        'distance': z_ij[i, j, 0, t].item(),
                        'approach_vel': z_ij[i, j, 1, t].item(),
                        'obstacle_size': z_ij[i, j, 2, t].item(),
                        'ttc': z_ij[i, j, 3, t].item(),
                    })
    
    # Sort by threat score
    high_threat_details.sort(key=lambda x: x['threat_score'], reverse=True)
    
    print(f"\nTop 20 High Threat Pairs:")
    print(f"{'Rank':<6} {'Ped (i)':<12} {'Type':<12} {'Obstacle (j)':<12} {'Type':<12} {'Frame':<8} {'Score':<8} {'Dist':<10} {'v+':<10} {'Size':<8} {'TTC':<8}")
    print("-" * 120)
    
    for rank, detail in enumerate(high_threat_details[:20], 1):
        print(f"{rank:<6} {detail['pedestrian_i']:<12} {detail['pedestrian_type_i']:<12} "
              f"{detail['obstacle_j']:<12} {detail['obstacle_type_j']:<12} "
              f"{detail['frame']:<8} {detail['threat_score']:<8.4f} "
              f"{detail['distance']:<10.2f} {detail['approach_vel']:<10.4f} "
              f"{detail['obstacle_size']:<8.2f} {detail['ttc']:<8.2f}")
    
    # Statistics by object type pairs
    print(f"\nHigh Threat Statistics by Object Type Pairs:")
    print(f"{'Pedestrian Type':<20} {'Obstacle Type':<20} {'Count':<10} {'Mean Score':<12} {'Max Score':<10}")
    print("-" * 80)
    
    for (type_i, type_j), scores in sorted(high_threat_by_type.items(), 
                                          key=lambda x: np.mean(x[1]), reverse=True):
        print(f"{type_i:<20} {type_j:<20} {len(scores):<10} {np.mean(scores):<12.4f} {np.max(scores):<10.4f}")
    
    # Analyze which obstacle types cause high threat
    obstacle_threat_counts = defaultdict(int)
    obstacle_threat_scores = defaultdict(list)
    
    for detail in high_threat_details:
        obs_type = detail['obstacle_type_j']
        obstacle_threat_counts[obs_type] += 1
        obstacle_threat_scores[obs_type].append(detail['threat_score'])
    
    print(f"\nHigh Threat by Obstacle Type:")
    print(f"{'Obstacle Type':<20} {'Count':<10} {'Mean Score':<12} {'Max Score':<10}")
    print("-" * 60)
    
    for obs_type in sorted(obstacle_threat_counts.keys(), 
                          key=lambda x: np.mean(obstacle_threat_scores[x]), reverse=True):
        print(f"{obs_type:<20} {obstacle_threat_counts[obs_type]:<10} "
              f"{np.mean(obstacle_threat_scores[obs_type]):<12.4f} "
              f"{np.max(obstacle_threat_scores[obs_type]):<10.4f}")
    
    return high_threat_details


if __name__ == "__main__":
    import os
    import sys
    
    scenes_to_test = [
        ('coupa', 'video0'),
        ('bookstore', 'video0'),
        ('deathCircle', 'video0'),
    ]
    
    if len(sys.argv) > 1:
        scene_name = sys.argv[1]
        video_id = sys.argv[2] if len(sys.argv) > 2 else 'video0'
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        analyze_high_threat(scene_name, video_id, threshold)
    else:
        for scene_name, video_id in scenes_to_test:
            try:
                analyze_high_threat(scene_name, video_id, threshold=0.4)
                print("\n")
            except Exception as e:
                print(f"Error testing {scene_name}/{video_id}: {e}")
                import traceback
                traceback.print_exc()
                print("\n")

