"""
Create filtered dataset based on motion types
"""

import os
import numpy as np
import shutil
from utils import TrajectoryDataset
from torch.utils.data import DataLoader
import argparse

def create_motion_filtered_dataset(source_dataset_path, target_dataset_path, 
                                 motion_types=['linear', 'curved', 'direction_change'],
                                 obs_len=8, pred_len=12):
    """
    Create a filtered dataset containing only specified motion types
    """
    
    # Create target directory
    os.makedirs(target_dataset_path, exist_ok=True)
    
    # Load source dataset
    dataset = TrajectoryDataset(source_dataset_path, obs_len=obs_len, pred_len=pred_len, skip=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    filtered_trajectories = []
    motion_counts = {motion_type: 0 for motion_type in motion_types}
    
    print(f"Filtering dataset for motion types: {motion_types}")
    print("=" * 50)
    
    for batch_idx, batch in enumerate(loader):
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = batch[:4]
        
        # Convert to numpy for analysis
        obs_traj = obs_traj.numpy()
        pred_traj = pred_traj.numpy()
        obs_traj_rel = obs_traj_rel.numpy()
        pred_traj_rel = pred_traj_rel.numpy()
        
        batch_size, num_ped, _, seq_len = obs_traj.shape
        
        for b in range(batch_size):
            for p in range(num_ped):
                # Get trajectory for this pedestrian
                traj = obs_traj[b, p, :, :]  # (2, obs_len)
                traj_rel = obs_traj_rel[b, p, :, :]  # (2, obs_len)
                
                # Skip if trajectory is empty
                if np.all(traj == 0):
                    continue
                
                # Analyze motion characteristics
                features = analyze_single_trajectory(traj, traj_rel)
                if features is None:
                    continue
                
                # Classify motion type
                motion_type = classify_motion_type(features)
                
                # Check if this motion type is in our filter
                if motion_type in motion_types:
                    # Store trajectory data
                    trajectory_data = {
                        'obs_traj': traj,
                        'pred_traj': pred_traj[b, p, :, :],
                        'obs_traj_rel': traj_rel,
                        'pred_traj_rel': pred_traj_rel[b, p, :, :],
                        'motion_type': motion_type,
                        'features': features
                    }
                    filtered_trajectories.append(trajectory_data)
                    motion_counts[motion_type] += 1
                    
                    if len(filtered_trajectories) % 100 == 0:
                        print(f"Found {len(filtered_trajectories)} trajectories so far...")
    
    # Save filtered trajectories
    save_filtered_trajectories(filtered_trajectories, target_dataset_path)
    
    print(f"\nFiltered dataset created with {len(filtered_trajectories)} trajectories")
    print("Motion type distribution:")
    for motion_type, count in motion_counts.items():
        percentage = (count / len(filtered_trajectories)) * 100 if len(filtered_trajectories) > 0 else 0
        print(f"  {motion_type}: {count} ({percentage:.1f}%)")
    
    return filtered_trajectories, motion_counts

def analyze_single_trajectory(traj, traj_rel):
    """
    Analyze a single trajectory and extract features
    """
    # Remove zero padding
    valid_mask = ~np.all(traj == 0, axis=0)
    traj = traj[:, valid_mask]
    traj_rel = traj_rel[:, valid_mask]
    
    if traj.shape[1] < 3:
        return None
    
    features = {}
    
    # 1. Linear vs Curved motion
    t = np.arange(traj.shape[1])
    poly_x = np.polyfit(t, traj[0, :], 2)
    poly_y = np.polyfit(t, traj[1, :], 2)
    
    # Calculate R-squared for linearity
    x_pred = np.polyval(poly_x, t)
    y_pred = np.polyval(poly_y, t)
    
    ss_res_x = np.sum((traj[0, :] - x_pred) ** 2)
    ss_tot_x = np.sum((traj[0, :] - np.mean(traj[0, :])) ** 2)
    r2_x = 1 - (ss_res_x / ss_tot_x) if ss_tot_x != 0 else 0
    
    ss_res_y = np.sum((traj[1, :] - y_pred) ** 2)
    ss_tot_y = np.sum((traj[1, :] - np.mean(traj[1, :])) ** 2)
    r2_y = 1 - (ss_res_y / ss_tot_y) if ss_tot_y != 0 else 0
    
    features['linearity'] = (r2_x + r2_y) / 2
    
    # 2. Speed analysis
    speeds = np.sqrt(traj_rel[0, :] ** 2 + traj_rel[1, :] ** 2)
    features['avg_speed'] = np.mean(speeds)
    features['max_speed'] = np.max(speeds)
    features['speed_variance'] = np.var(speeds)
    
    # 3. Direction changes
    directions = np.arctan2(traj_rel[1, :], traj_rel[0, :])
    direction_changes = np.abs(np.diff(directions))
    direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)
    features['direction_change_rate'] = np.mean(direction_changes)
    features['max_direction_change'] = np.max(direction_changes)
    
    # 4. Acceleration
    if len(speeds) > 1:
        acceleration = np.diff(speeds)
        features['avg_acceleration'] = np.mean(np.abs(acceleration))
        features['max_acceleration'] = np.max(np.abs(acceleration))
    else:
        features['avg_acceleration'] = 0
        features['max_acceleration'] = 0
    
    # 5. Trajectory length
    total_distance = np.sum(speeds)
    features['total_distance'] = total_distance
    
    # 6. Displacement
    displacement = np.sqrt((traj[0, -1] - traj[0, 0]) ** 2 + (traj[1, -1] - traj[1, 0]) ** 2)
    features['displacement'] = displacement
    
    # 7. Efficiency (displacement / total_distance)
    features['efficiency'] = displacement / total_distance if total_distance > 0 else 0
    
    return features

def classify_motion_type(features):
    """
    Classify motion type based on features
    """
    if features is None:
        return 'stationary'
    
    # Stationary motion
    if features['avg_speed'] < 0.1:
        return 'stationary'
    
    # Linear motion
    if features['linearity'] > 0.9 and features['direction_change_rate'] < 0.3:
        return 'linear'
    
    # Curved motion
    if features['linearity'] < 0.7:
        return 'curved'
    
    # Direction change motion
    if features['direction_change_rate'] > 0.5 or features['max_direction_change'] > 1.0:
        return 'direction_change'
    
    # Group motion (high speed, low direction change)
    if features['avg_speed'] > 0.5 and features['direction_change_rate'] < 0.2:
        return 'group_motion'
    
    return 'linear'  # default

def save_filtered_trajectories(trajectories, target_path):
    """
    Save filtered trajectories in the same format as original dataset
    """
    # Group trajectories by motion type
    motion_groups = {}
    for traj in trajectories:
        motion_type = traj['motion_type']
        if motion_type not in motion_groups:
            motion_groups[motion_type] = []
        motion_groups[motion_type].append(traj)
    
    # Save each motion type as a separate file
    for motion_type, traj_list in motion_groups.items():
        filename = f"{motion_type}_filtered.txt"
        filepath = os.path.join(target_path, filename)
        
        with open(filepath, 'w') as f:
            for i, traj in enumerate(traj_list):
                # Write observation trajectory
                obs_traj = traj['obs_traj']
                for t in range(obs_traj.shape[1]):
                    f.write(f"{t}\t{i}\t{obs_traj[0, t]:.4f}\t{obs_traj[1, t]:.4f}\n")
                
                # Write prediction trajectory
                pred_traj = traj['pred_traj']
                for t in range(pred_traj.shape[1]):
                    f.write(f"{t + obs_traj.shape[1]}\t{i}\t{pred_traj[0, t]:.4f}\t{pred_traj[1, t]:.4f}\n")
        
        print(f"Saved {len(traj_list)} {motion_type} trajectories to {filename}")

def create_motion_specific_datasets(source_dataset_path, target_base_path, 
                                  motion_combinations=None, obs_len=8, pred_len=12):
    """
    Create multiple filtered datasets for different motion combinations
    """
    if motion_combinations is None:
        motion_combinations = {
            'linear_only': ['linear'],
            'curved_only': ['curved'],
            'direction_change_only': ['direction_change'],
            'group_motion_only': ['group_motion'],
            'linear_curved': ['linear', 'curved'],
            'all_motions': ['linear', 'curved', 'direction_change', 'group_motion']
        }
    
    results = {}
    
    for dataset_name, motion_types in motion_combinations.items():
        target_path = os.path.join(target_base_path, dataset_name)
        print(f"\nCreating {dataset_name} dataset...")
        print(f"Motion types: {motion_types}")
        
        trajectories, counts = create_motion_filtered_dataset(
            source_dataset_path, target_path, motion_types, obs_len, pred_len
        )
        
        results[dataset_name] = {
            'trajectories': trajectories,
            'counts': counts,
            'path': target_path
        }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', default='eth', help='Source dataset name')
    parser.add_argument('--source_split', default='train', help='Source data split')
    parser.add_argument('--target_base_path', default='./datasets_filtered/', help='Target base path')
    parser.add_argument('--motion_types', nargs='+', default=['linear', 'curved', 'direction_change'],
                       help='Motion types to include')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--create_all_combinations', action='store_true',
                       help='Create all possible motion combinations')
    
    args = parser.parse_args()
    
    source_dataset_path = f'./datasets/{args.source_dataset}/{args.source_split}/'
    
    if args.create_all_combinations:
        # Create all possible combinations
        results = create_motion_specific_datasets(
            source_dataset_path, args.target_base_path, 
            obs_len=args.obs_len, pred_len=args.pred_len
        )
        
        print("\n" + "="*50)
        print("SUMMARY OF CREATED DATASETS")
        print("="*50)
        for dataset_name, result in results.items():
            print(f"\n{dataset_name}:")
            print(f"  Path: {result['path']}")
            print(f"  Total trajectories: {len(result['trajectories'])}")
            for motion_type, count in result['counts'].items():
                print(f"    {motion_type}: {count}")
    else:
        # Create single filtered dataset
        target_path = os.path.join(args.target_base_path, f"{args.source_dataset}_{args.source_split}_filtered")
        trajectories, counts = create_motion_filtered_dataset(
            source_dataset_path, target_path, args.motion_types, 
            args.obs_len, args.pred_len
        )
        
        print(f"\nFiltered dataset created at: {target_path}")

if __name__ == "__main__":
    main()
