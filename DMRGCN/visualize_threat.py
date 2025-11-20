"""
Threat Score Visualization Script
특정 인물에 대한 threat score를 영상에 시각화하는 스크립트

Usage:
    python visualize_threat.py \
        --dataset bookstore \
        --dataset_type sdd \
        --target_track_id 5 \
        --sequence_idx 0 \
        --output_path ./threat_viz.mp4 \
        --labels_dir /path/to/labels \
        --class_size_csv /path/to/class_sizes.csv
"""

import os
import sys
import argparse
import torch
import pickle
from torch.utils.data import DataLoader

from model import social_dmrgcn
from utils import SDDTrajectoryDataset
from utils.threat_visualizer import (
    create_threat_visualization_video,
    save_threat_visualization_frames,
    visualize_threat_scores_for_person
)


def load_model(checkpoint_dir, dataset_name, device='cuda:0'):
    """Load trained model from checkpoint."""
    args_path = os.path.join(checkpoint_dir, 'args.pkl')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Args file not found: {args_path}")
    
    with open(args_path, 'rb') as f:
        train_args = pickle.load(f)
    
    model_path = os.path.join(checkpoint_dir, f'{dataset_name}_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, f'{dataset_name}.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = social_dmrgcn(
        n_stgcn=train_args.n_stgcn,
        n_tpcnn=train_args.n_tpcnn,
        input_feat=train_args.input_size,
        output_feat=train_args.output_size,
        seq_len=train_args.obs_seq_len,
        pred_seq_len=train_args.pred_seq_len,
        kernel_size=train_args.kernel_size
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, train_args


def get_track_ids_from_dataset(dataset, sequence_idx):
    """Extract track IDs from dataset sequence.
    
    This function gets track IDs from the dataset, which now stores them.
    """
    if hasattr(dataset, 'get_track_ids'):
        return dataset.get_track_ids(sequence_idx)
    else:
        # Fallback: try to get from batch
        batch = dataset[sequence_idx]
        if len(batch) > 10:
            return batch[10]
        else:
            # Last resort: use indices
            obs_traj = batch[0]
            num_tracks = obs_traj.shape[0]
            return list(range(num_tracks))


def main():
    parser = argparse.ArgumentParser(description='Visualize threat scores for specific person')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., bookstore)')
    parser.add_argument('--dataset_type', type=str, default='sdd', choices=['generic', 'sdd'],
                       help='Dataset type')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: ./sdd_datasets/<dataset>/val/)')
    parser.add_argument('--labels_dir', type=str, default=None,
                       help='Directory with <video>_labels.csv files')
    parser.add_argument('--class_size_csv', type=str, default=None,
                       help='CSV with class sizes: label,size')
    
    # Model parameters
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory (default: ./checkpoints/<tag>/)')
    parser.add_argument('--tag', type=str, default='dmrgcn_4rel_sdd',
                       help='Model tag (experiment name)')
    
    # Visualization parameters
    parser.add_argument('--target_track_id', type=int, required=True,
                       help='Track ID of target person to visualize')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Sequence index in dataset')
    parser.add_argument('--output_path', type=str, default='./threat_visualization.mp4',
                       help='Output video path')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for frames (if saving frames instead of video)')
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to original video file (optional)')
    parser.add_argument('--scale', type=float, default=1000.0,
                       help='Scale factor for coordinates (1000.0 for normalized SDD)')
    parser.add_argument('--visualize_mode', type=str, default='all',
                       choices=['arrows', 'circles', 'heatmap', 'all'],
                       help='Visualization mode')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual frames instead of video')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Output video frame rate')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set up paths
    if args.data_dir is None:
        if args.dataset_type == 'sdd':
            args.data_dir = f'./sdd_datasets/{args.dataset}/val/'
        else:
            args.data_dir = f'./datasets/{args.dataset}/val/'
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'./checkpoints/{args.tag}/'
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"Warning: Checkpoint directory not found: {args.checkpoint_dir}")
        print("Will visualize threat scores without model inference (using computed A_obs)")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    if args.dataset_type == 'sdd':
        dataset = SDDTrajectoryDataset(
            args.data_dir,
            obs_len=8,  # Default, will be updated from model args if available
            pred_len=12,
            skip=1,
            labels_dir=args.labels_dir,
            class_size_csv=args.class_size_csv
        )
    else:
        from utils import TrajectoryDataset
        dataset = TrajectoryDataset(
            args.data_dir,
            obs_len=8,
            pred_len=12,
            skip=1
        )
    
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Check sequence index
    if args.sequence_idx >= len(dataset):
        print(f"Error: Sequence index {args.sequence_idx} is out of range (max: {len(dataset)-1})")
        sys.exit(1)
    
    # Load model if checkpoint exists
    model = None
    train_args = None
    if os.path.exists(args.checkpoint_dir):
        try:
            print(f"Loading model from: {args.checkpoint_dir}")
            model, train_args = load_model(args.checkpoint_dir, args.dataset, args.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Will visualize threat scores from dataset only (A_obs)")
    
    # Get track IDs
    track_ids = get_track_ids_from_dataset(dataset, args.sequence_idx)
    print(f"Available track IDs in sequence {args.sequence_idx}: {track_ids}")
    
    # Check if target track ID exists
    if args.target_track_id not in track_ids:
        print(f"Warning: Target track ID {args.target_track_id} not found in sequence")
        print(f"Available track IDs: {track_ids}")
        print(f"Using track ID {track_ids[0]} instead")
        args.target_track_id = track_ids[0]
    
    # Create visualization
    if args.save_frames:
        output_dir = args.output_dir or args.output_path.replace('.mp4', '_frames')
        print(f"Saving frames to: {output_dir}")
        save_threat_visualization_frames(
            dataset=dataset,
            model=model,
            target_track_id=args.target_track_id,
            sequence_idx=args.sequence_idx,
            output_dir=output_dir,
            video_path=args.video_path,
            scale=args.scale,
            visualize_mode=args.visualize_mode,
            device=args.device
        )
    else:
        print(f"Creating video: {args.output_path}")
        create_threat_visualization_video(
            dataset=dataset,
            model=model,
            target_track_id=args.target_track_id,
            sequence_idx=args.sequence_idx,
            output_path=args.output_path,
            video_path=args.video_path,
            scale=args.scale,
            visualize_mode=args.visualize_mode,
            fps=args.fps,
            device=args.device
        )
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()

