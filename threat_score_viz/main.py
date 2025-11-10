#!/usr/bin/env python3
"""
Main CLI script for threat score visualization.

This script processes SDD videos and annotations to compute and visualize
threat scores for pedestrian trajectories.
"""

import argparse
import os
import sys
from threat_score_viz.visualizer import process_video_with_threat_scores
from threat_score_viz.video_utils import verify_annotation_video_alignment
from threat_score_viz.target_selector import find_best_target_candidates, select_target


def main():
    """Main entry point for the threat score visualization CLI."""
    parser = argparse.ArgumentParser(
        description='Compute and visualize threat scores for SDD dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire video with auto-selected target
  python -m threat_score_viz.main \\
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \\
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \\
    --output-video output/annotated_video.mp4 \\
    --output-metadata output/metadata.json

  # Process specific frame range with custom target
  python -m threat_score_viz.main \\
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \\
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \\
    --output-video output/annotated_video.mp4 \\
    --output-metadata output/metadata.json \\
    --target-id 176 \\
    --start-frame 2500 \\
    --end-frame 3000

  # List available target candidates
  python -m threat_score_viz.main \\
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \\
    --list-candidates
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Path to annotation file'
    )
    parser.add_argument(
        '--output-video',
        type=str,
        help='Path to save annotated video output'
    )
    parser.add_argument(
        '--output-metadata',
        type=str,
        help='Path to save metadata JSON file'
    )
    parser.add_argument(
        '--target-id',
        type=int,
        default=None,
        help='Target pedestrian ID (if not provided, will auto-select)'
    )
    parser.add_argument(
        '--list-candidates',
        action='store_true',
        help='List available target candidates and exit'
    )
    parser.add_argument(
        '--prefer-center',
        action='store_true',
        default=True,
        help='Prefer centrally located targets when auto-selecting (default: True)'
    )
    parser.add_argument(
        '--no-prefer-center',
        action='store_true',
        help='Do not prefer centrally located targets (use frame count only)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame (default: 0)'
    )
    parser.add_argument(
        '--end-frame',
        type=int,
        default=None,
        help='Ending frame (default: process all frames)'
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=('W1', 'W2', 'W3', 'W4'),
        help='Weights for threat score features (default: 1.0 1.0 1.0 1.0)'
    )
    parser.add_argument(
        '--max-distance',
        type=float,
        default=100.0,
        help='Maximum distance for normalization (default: 100.0)'
    )
    parser.add_argument(
        '--max-vel-diff',
        type=float,
        default=10.0,
        help='Maximum velocity difference for normalization (default: 10.0)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor for text size (default: 1.0)'
    )
    parser.add_argument(
        '--no-target-marker',
        action='store_true',
        help='Do not draw target marker'
    )
    parser.add_argument(
        '--no-graph',
        action='store_true',
        help='Disable graph visualization (use simple annotations instead)'
    )
    parser.add_argument(
        '--no-edges',
        action='store_true',
        help='Do not draw edges between target and obstacles'
    )
    parser.add_argument(
        '--no-nodes',
        action='store_true',
        help='Do not draw nodes for obstacles'
    )
    parser.add_argument(
        '--no-coord-transform',
        action='store_true',
        help='Do not apply coordinate transformation (use annotations as pixel coordinates directly)'
    )
    parser.add_argument(
        '--verify-alignment',
        action='store_true',
        help='Verify video-annotation alignment before processing'
    )
    
    args = parser.parse_args()
    
    # List candidates mode
    if args.list_candidates:
        print("Finding target candidates...")
        
        # Try to get video dimensions if video is provided
        video_width = None
        video_height = None
        if args.video:
            try:
                from .video_utils import get_video_properties
                video_props = get_video_properties(args.video)
                video_width = video_props['width']
                video_height = video_props['height']
                print(f"Video dimensions: {video_width}x{video_height}")
            except Exception as e:
                print(f"Warning: Could not get video dimensions: {e}")
        
        # If video dimensions available, show central candidates
        if video_width and video_height and args.prefer_center:
            from .target_selector import find_central_target_candidates
            candidates = find_central_target_candidates(
                args.annotations,
                video_width,
                video_height,
                min_frames=100,
                max_candidates=20
            )
            print(f"\nFound {len(candidates)} central candidates (min 100 frames, relative to object cluster center):\n")
            if candidates:
                center_used = candidates[0][1].get('center_used', (0, 0))
                print(f"Cluster center used: ({center_used[0]:.1f}, {center_used[1]:.1f})")
            print(f"{'Rank':<6} {'Object ID':<12} {'Frames':<10} {'Avg Position':<20} {'Dist from Center':<18} {'Score':<8}")
            print("-" * 80)
            for i, (obj_id, stat) in enumerate(candidates, 1):
                avg_pos = stat.get('avg_position', (0, 0))
                dist = stat.get('distance_from_center', 0)
                score = stat.get('combined_score', 0)
                pos_str = f"({avg_pos[0]:.1f}, {avg_pos[1]:.1f})"
                dist_str = f"{dist:.1f}px"
                print(f"{i:<6} {obj_id:<12} {stat['frame_count']:<10} {pos_str:<20} {dist_str:<18} {score:.3f}")
        else:
            from .target_selector import find_best_target_candidates
            candidates = find_best_target_candidates(
                args.annotations,
                min_frames=100,
                max_candidates=20
            )
            print(f"\nFound {len(candidates)} candidates (min 100 frames):\n")
            print(f"{'Rank':<6} {'Object ID':<12} {'Frames':<10} {'Frame Range':<20}")
            print("-" * 50)
            for i, (obj_id, stat) in enumerate(candidates, 1):
                frame_range = f"{stat['first_frame']}-{stat['last_frame']}"
                print(f"{i:<6} {obj_id:<12} {stat['frame_count']:<10} {frame_range:<20}")
        return
    
    # Validate required arguments for processing
    if not args.video:
        parser.error("--video is required for processing (use --list-candidates to list targets)")
    
    if not args.output_video and not args.output_metadata:
        parser.error("At least one of --output-video or --output-metadata must be provided")
    
    # Verify alignment if requested
    if args.verify_alignment:
        print("Verifying video-annotation alignment...")
        is_aligned, info = verify_annotation_video_alignment(
            args.annotations,
            args.video
        )
        if not is_aligned:
            print("WARNING: Video and annotation frame counts do not match!")
            print(f"  Video frames: {info['video_frame_count']}")
            print(f"  Annotation frames: {info['annotation_frame_count']}")
            print(f"  Frame range: {info['annotation_min_frame']}-{info['annotation_max_frame']}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        else:
            print("✓ Alignment verified")
    
    # Create output directory if needed
    if args.output_video:
        output_dir = os.path.dirname(args.output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
    
    if args.output_metadata:
        output_dir = os.path.dirname(args.output_metadata)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
    
    # Process video
    print("\nProcessing video...")
    print(f"  Video: {args.video}")
    print(f"  Annotations: {args.annotations}")
    if args.output_video:
        print(f"  Output video: {args.output_video}")
    if args.output_metadata:
        print(f"  Output metadata: {args.output_metadata}")
    print(f"  Target ID: {args.target_id if args.target_id else 'Auto-select'}")
    print(f"  Frame range: {args.start_frame} to {args.end_frame if args.end_frame else 'end'}")
    print(f"  Weights: {args.weights}")
    
    try:
        stats = process_video_with_threat_scores(
            video_path=args.video,
            annotation_path=args.annotations,
            output_video_path=args.output_video or '/dev/null',
            output_metadata_path=args.output_metadata or '/dev/null',
            target_id=args.target_id,
            auto_select_target=args.target_id is None,
            weights=tuple(args.weights),
            max_distance=args.max_distance,
            max_vel_diff=args.max_vel_diff,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            scale=args.scale,
            draw_target=not args.no_target_marker,
            draw_graph=not args.no_graph,
            draw_edges=not args.no_edges,
            draw_nodes=not args.no_nodes,
            apply_coordinate_transform=not args.no_coord_transform
        )
        
        # Note about obstacles
        print(f"\nNote: All other objects in each frame are treated as obstacles/threats.")
        print(f"      The target person (ID: {stats.get('target_id', 'N/A')}) is the center of the graph.")
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Frames skipped: {stats['frames_skipped']}")
        print(f"Total frames: {stats['total_frames']}")
        if args.output_video:
            print(f"Output video: {stats['output_video']}")
        if args.output_metadata:
            print(f"Output metadata: {stats['output_metadata']}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

