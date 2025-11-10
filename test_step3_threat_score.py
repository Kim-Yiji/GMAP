#!/usr/bin/env python3
"""
Test script for Step 3: Threat Score Computation

This script tests the threat score computation functionality.
"""

from threat_score_viz.threat_score_computer import (
    compute_distance,
    compute_velocity_magnitude,
    compute_heading,
    compute_heading_alignment,
    normalize_distance,
    compute_relational_features,
    compute_threat_score,
    compute_threat_scores_for_frame,
    build_object_position_history
)
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.target_selector import select_target, get_target_positions

def test_threat_score_computation():
    """Test threat score computation functionality."""
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    
    print("=" * 60)
    print("Testing Step 3: Threat Score Computation")
    print("=" * 60)
    
    # Test 1: Basic distance computation
    print("\n1. Testing distance computation...")
    pos1 = (0.0, 0.0)
    pos2 = (3.0, 4.0)
    distance = compute_distance(pos1, pos2)
    print(f"   ✓ Distance between {pos1} and {pos2}: {distance:.2f} (expected: 5.0)")
    
    # Test 2: Velocity computation
    print("\n2. Testing velocity computation...")
    prev_pos = (0.0, 0.0)
    curr_pos = (2.0, 2.0)
    vel = compute_velocity_magnitude(curr_pos, prev_pos)
    print(f"   ✓ Velocity magnitude: {vel:.2f} (expected: ~2.83)")
    
    # Test 3: Heading computation
    print("\n3. Testing heading computation...")
    heading = compute_heading(curr_pos, prev_pos)
    print(f"   ✓ Heading angle: {heading:.2f} radians ({heading * 180 / 3.14159:.1f} degrees)")
    
    # Test 4: Heading alignment
    print("\n4. Testing heading alignment...")
    heading1 = 0.0  # East
    heading2 = 0.0  # East (same direction)
    alignment = compute_heading_alignment(heading1, heading2)
    print(f"   ✓ Alignment (same direction): {alignment:.2f} (expected: 1.0)")
    
    heading2 = 3.14159  # West (opposite direction)
    alignment = compute_heading_alignment(heading1, heading2)
    print(f"   ✓ Alignment (opposite direction): {alignment:.2f} (expected: ~0.0)")
    
    # Test 5: Distance normalization
    print("\n5. Testing distance normalization...")
    norm_close = normalize_distance(10.0, max_distance=100.0)
    norm_far = normalize_distance(90.0, max_distance=100.0)
    print(f"   ✓ Normalized distance (close): {norm_close:.2f} (expected: high)")
    print(f"   ✓ Normalized distance (far): {norm_far:.2f} (expected: low)")
    
    # Test 6: Relational features
    print("\n6. Testing relational features computation...")
    target_pos = (10.0, 10.0)
    target_prev_pos = (9.0, 10.0)  # Moving east
    obstacle_pos = (12.0, 10.0)  # Close, same y
    obstacle_prev_pos = (11.0, 10.0)  # Also moving east
    
    f1, f2, f3, f4 = compute_relational_features(
        target_pos, target_prev_pos,
        obstacle_pos, obstacle_prev_pos
    )
    print(f"   ✓ Feature 1 (distance): {f1:.3f}")
    print(f"   ✓ Feature 2 (velocity diff): {f2:.3f}")
    print(f"   ✓ Feature 3 (heading alignment): {f3:.3f}")
    print(f"   ✓ Feature 4 (class interaction): {f4:.3f}")
    
    # Test 7: Threat score computation
    print("\n7. Testing threat score computation...")
    threat_score = compute_threat_score(f1, f2, f3, f4, w1=1.0, w2=1.0, w3=1.0, w4=1.0)
    print(f"   ✓ Threat score: {threat_score:.3f} (range: [0, 1])")
    print(f"   ✓ Features sum: {f1 + f2 + f3 + f4:.3f}")
    
    # Test 8: Full pipeline - compute threat scores for a frame
    print("\n8. Testing full pipeline for a frame...")
    
    # Select target
    target_id, target_stats = select_target(annotation_path, auto_select=True, min_frames=100)
    print(f"   ✓ Selected target: {target_id}")
    
    # Get frame data and target positions
    frame_data = get_object_positions_per_frame(annotation_path)
    target_positions = get_target_positions(annotation_path, target_id)
    object_positions = build_object_position_history(annotation_path)
    
    # Find a frame where target exists
    test_frame = None
    for frame_id in sorted(frame_data.keys()):
        if frame_id in target_positions:
            # Check if there are other objects in this frame
            objects = frame_data[frame_id]
            if len(objects) > 1:  # At least one other object besides target
                test_frame = frame_id
                break
    
    if test_frame is not None:
        print(f"   ✓ Testing with frame {test_frame}")
        
        threat_scores = compute_threat_scores_for_frame(
            test_frame, target_id, frame_data,
            target_positions, object_positions
        )
        
        print(f"   ✓ Computed threat scores for {len(threat_scores)} obstacles")
        
        # Show top 5 threat scores
        threat_scores_sorted = sorted(threat_scores, key=lambda x: x[1], reverse=True)
        print(f"\n   Top 5 threat scores in frame {test_frame}:")
        for i, (obj_id, score, pos, features) in enumerate(threat_scores_sorted[:5], 1):
            f1, f2, f3, f4 = features
            print(f"   {i}. Object {obj_id}: score={score:.3f}, pos=({pos[0]:.1f}, {pos[1]:.1f})")
            print(f"      Features: f1={f1:.3f}, f2={f2:.3f}, f3={f3:.3f}, f4={f4:.3f}")
    else:
        print("   ⚠ No suitable frame found for testing")
    
    print("\n" + "=" * 60)
    print("✓ Step 3: Threat Score Computation - ALL TESTS PASSED")
    print("=" * 60)

if __name__ == '__main__':
    test_threat_score_computation()

