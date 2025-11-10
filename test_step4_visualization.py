#!/usr/bin/env python3
"""
Test script for Step 4: Visualization

This script tests the visualization functionality with a small sample of frames.
"""

import os
import cv2
from threat_score_viz.visualizer import (
    draw_threat_score_on_frame,
    draw_target_marker,
    process_video_frame,
    create_frame_metadata,
    process_video_with_threat_scores
)
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.target_selector import select_target, get_target_positions
from threat_score_viz.threat_score_computer import (
    compute_threat_scores_for_frame,
    build_object_position_history
)

def test_single_frame_visualization():
    """Test visualization on a single frame."""
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    video_path = 'sdd_bookstore/bookstore_vid/video0/video.mp4'
    
    print("=" * 60)
    print("Testing Step 4: Visualization (Single Frame)")
    print("=" * 60)
    
    # Select target
    target_id, target_stats = select_target(annotation_path, auto_select=True, min_frames=100)
    print(f"\n1. Selected target: {target_id}")
    
    # Load data
    frame_data = get_object_positions_per_frame(annotation_path)
    target_positions = get_target_positions(annotation_path, target_id)
    object_positions = build_object_position_history(annotation_path)
    
    # Find a good test frame
    test_frame_id = None
    for frame_id in sorted(frame_data.keys()):
        if frame_id in target_positions and frame_id >= 2500:
            objects = frame_data[frame_id]
            if len(objects) > 1:
                test_frame_id = frame_id
                break
    
    if test_frame_id is None:
        print("   ⚠ No suitable test frame found")
        return
    
    print(f"2. Testing with frame {test_frame_id}")
    
    # Load frame from video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_id)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("   ⚠ Could not load frame from video")
        return
    
    print(f"   ✓ Loaded frame: {frame.shape}")
    
    # Compute threat scores
    threat_scores = compute_threat_scores_for_frame(
        test_frame_id, target_id, frame_data,
        target_positions, object_positions
    )
    print(f"   ✓ Computed {len(threat_scores)} threat scores")
    
    # Get target position
    target_position = target_positions[test_frame_id]
    
    # Process frame
    annotated_frame = process_video_frame(
        frame, test_frame_id, target_id, target_position,
        threat_scores, draw_target=True, scale=1.0
    )
    print(f"   ✓ Processed annotated frame: {annotated_frame.shape}")
    
    # Save test frame
    output_dir = 'test_output'
    os.makedirs(output_dir, exist_ok=True)
    test_output_path = os.path.join(output_dir, 'test_frame_annotated.png')
    cv2.imwrite(test_output_path, annotated_frame)
    print(f"   ✓ Saved test frame to: {test_output_path}")
    
    # Test metadata creation
    frame_metadata = create_frame_metadata(
        test_frame_id, target_id, target_position, threat_scores
    )
    print(f"\n3. Frame metadata:")
    print(f"   - Frame ID: {frame_metadata['frame_id']}")
    print(f"   - Target ID: {frame_metadata['target_id']}")
    print(f"   - Target position: {frame_metadata['target_position']}")
    print(f"   - Number of interactions: {len(frame_metadata['interactions'])}")
    if frame_metadata['interactions']:
        top_interaction = max(frame_metadata['interactions'], key=lambda x: x['score'])
        print(f"   - Top threat: Object {top_interaction['object_id']} with score {top_interaction['score']:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ Step 4: Visualization (Single Frame) - TEST PASSED")
    print("=" * 60)

def test_video_processing_sample():
    """Test processing a small sample of video frames."""
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    video_path = 'sdd_bookstore/bookstore_vid/video0/video.mp4'
    
    print("\n" + "=" * 60)
    print("Testing Step 4: Video Processing (Sample)")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    output_video_path = os.path.join(output_dir, 'test_video_sample.mp4')
    output_metadata_path = os.path.join(output_dir, 'test_metadata_sample.json')
    
    print(f"\nProcessing sample video (frames 2500-2510)...")
    
    try:
        stats = process_video_with_threat_scores(
            video_path=video_path,
            annotation_path=annotation_path,
            output_video_path=output_video_path,
            output_metadata_path=output_metadata_path,
            target_id=None,
            auto_select_target=True,
            start_frame=2500,
            end_frame=2510,  # Just 11 frames for testing
            scale=1.0,
            draw_target=True
        )
        
        print(f"\n✓ Processing complete!")
        print(f"   - Frames processed: {stats['frames_processed']}")
        print(f"   - Frames skipped: {stats['frames_skipped']}")
        print(f"   - Output video: {stats['output_video']}")
        print(f"   - Output metadata: {stats['output_metadata']}")
        
        # Check if files were created
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / 1024  # KB
            print(f"   - Video file size: {file_size:.1f} KB")
        
        if os.path.exists(output_metadata_path):
            file_size = os.path.getsize(output_metadata_path) / 1024  # KB
            print(f"   - Metadata file size: {file_size:.1f} KB")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✓ Step 4: Video Processing (Sample) - TEST PASSED")
    print("=" * 60)

if __name__ == '__main__':
    test_single_frame_visualization()
    test_video_processing_sample()

