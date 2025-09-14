"""
ETH/UCY 데이터셋 모션 분석 스크립트
데이터셋에서 다양한 보행자 모션 타입을 분석합니다.

이 스크립트는 미팅에서 요청된 "어떤 모션을 핸들링하고자 하는지" 파악하기 위해
데이터셋의 보행자 궤적들을 분석하여 모션 타입별 분포를 제공합니다.

분석하는 모션 타입:
1. Linear Motion: 직선 궤적, 최소한의 방향 변화
2. Curved Motion: 비선형 궤적, 부드러운 곡선
3. Direction Change: 상당한 방향 변화가 있는 궤적
4. Group Motion: 고속, 조율된 그룹 움직임
5. Stationary: 최소한의 움직임 궤적

분석 방법:
- 궤적의 선형성 (R-squared 값)
- 속도 분포 (평균, 최대, 분산)
- 방향 변화율
- 가속도
- 궤적 효율성 (변위/총 거리)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import TrajectoryDataset  # DMRGCN의 데이터 로더 사용
from torch.utils.data import DataLoader
import argparse

def analyze_trajectory_motions(dataset_path, obs_len=8, pred_len=12):
    """
    데이터셋에서 다양한 모션 타입을 분석하는 함수
    
    Args:
        dataset_path: 데이터셋 경로
        obs_len: 관찰 시퀀스 길이
        pred_len: 예측 시퀀스 길이
    
    Returns:
        motion_stats: 각 모션 타입별 통계
        trajectory_features: 각 궤적의 특징들
    """
    dataset = TrajectoryDataset(dataset_path, obs_len=obs_len, pred_len=pred_len, skip=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    motion_stats = {
        'linear': 0,
        'curved': 0,
        'stationary': 0,
        'direction_change': 0,
        'group_motion': 0,
        'total_trajectories': 0
    }
    
    trajectory_features = []
    
    print("Analyzing trajectory motions...")
    
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
                
                motion_stats['total_trajectories'] += 1
                
                # Analyze motion characteristics
                features = analyze_single_trajectory(traj, traj_rel)
                trajectory_features.append(features)
                
                # Classify motion type
                motion_type = classify_motion_type(features)
                motion_stats[motion_type] += 1
                
                if batch_idx % 100 == 0 and p == 0:
                    print(f"Processed batch {batch_idx}, trajectory {motion_stats['total_trajectories']}")
    
    return motion_stats, trajectory_features

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
    # Fit polynomial to trajectory
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

def plot_motion_distribution(motion_stats, save_path=None):
    """
    Plot distribution of motion types
    """
    motion_types = list(motion_stats.keys())
    counts = list(motion_stats.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(motion_types, counts)
    plt.title('Distribution of Motion Types in Dataset')
    plt.xlabel('Motion Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_distribution(trajectory_features, feature_name, save_path=None):
    """
    Plot distribution of a specific feature
    """
    if not trajectory_features:
        print("No trajectory features available")
        return
    
    valid_features = [f[feature_name] for f in trajectory_features if f is not None and feature_name in f]
    
    if not valid_features:
        print(f"Feature {feature_name} not found in trajectory features")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(valid_features, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='eth', help='Dataset name')
    parser.add_argument('--data_split', default='train', help='Data split (train/val/test)')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    dataset_path = f'./datasets/{args.dataset}/{args.data_split}/'
    
    print(f"Analyzing motions in {dataset_path}")
    print("=" * 50)
    
    # Analyze motions
    motion_stats, trajectory_features = analyze_trajectory_motions(
        dataset_path, args.obs_len, args.pred_len
    )
    
    # Print results
    print("\nMotion Statistics:")
    print("-" * 30)
    for motion_type, count in motion_stats.items():
        percentage = (count / motion_stats['total_trajectories']) * 100 if motion_stats['total_trajectories'] > 0 else 0
        print(f"{motion_type}: {count} ({percentage:.1f}%)")
    
    # Plot motion distribution
    plot_path = f'motion_distribution_{args.dataset}_{args.data_split}.png' if args.save_plots else None
    plot_motion_distribution(motion_stats, plot_path)
    
    # Plot feature distributions
    if trajectory_features:
        features_to_plot = ['linearity', 'avg_speed', 'direction_change_rate', 'efficiency']
        for feature in features_to_plot:
            plot_path = f'{feature}_distribution_{args.dataset}_{args.data_split}.png' if args.save_plots else None
            plot_feature_distribution(trajectory_features, feature, plot_path)
    
    print(f"\nAnalysis complete! Total trajectories: {motion_stats['total_trajectories']}")

if __name__ == "__main__":
    main()
