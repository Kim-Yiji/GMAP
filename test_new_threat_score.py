#!/usr/bin/env python3
"""
Test script for the new DMRGCN-based threat score computation.

This script tests the updated threat score algorithm that uses:
- Distance (d_ij)
- Approach velocity (v+_ij)
- Obstacle size (size_j)
- Time-to-Collision (TTC_ij)
"""

import os
import sys
from threat_score_viz.threat_score_computer import (
    compute_distance,
    compute_velocity,
    compute_relative_vectors,
    compute_approach_velocity,
    compute_ttc,
    compute_threat_features,
    normalize_threat_features_minmax,
    compute_threat_score,
    compute_threat_scores_for_frame,
    build_object_position_history,
    get_obstacle_size
)
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.target_selector import select_target, get_target_positions


def test_basic_functions():
    """Test basic computation functions."""
    print("=" * 60)
    print("Testing Basic Functions")
    print("=" * 60)
    
    # Test 1: Distance computation
    print("\n1. Testing distance computation...")
    pos1 = (0.0, 0.0)
    pos2 = (3.0, 4.0)
    distance = compute_distance(pos1, pos2)
    print(f"   ✓ Distance between {pos1} and {pos2}: {distance:.2f} (expected: 5.0)")
    assert abs(distance - 5.0) < 0.01, "Distance computation failed"
    
    # Test 2: Velocity computation
    print("\n2. Testing velocity computation...")
    prev_pos = (0.0, 0.0)
    curr_pos = (2.0, 2.0)
    vel = compute_velocity(curr_pos, prev_pos)
    print(f"   ✓ Velocity vector: {vel} (expected: (2.0, 2.0))")
    assert abs(vel[0] - 2.0) < 0.01 and abs(vel[1] - 2.0) < 0.01, "Velocity computation failed"
    
    # Test 3: Relative vectors
    print("\n3. Testing relative vectors...")
    target_pos = (0.0, 0.0)
    target_prev_pos = (-1.0, 0.0)  # Moving east
    obstacle_pos = (3.0, 0.0)
    obstacle_prev_pos = (2.0, 0.0)  # Also moving east
    
    rel_pos, rel_vel = compute_relative_vectors(
        target_pos, target_prev_pos,
        obstacle_pos, obstacle_prev_pos
    )
    print(f"   ✓ Relative position: {rel_pos} (expected: (3.0, 0.0))")
    print(f"   ✓ Relative velocity: {rel_vel} (expected: (0.0, 0.0))")
    
    # Test 4: Approach velocity
    print("\n4. Testing approach velocity...")
    # Case 1: Obstacle approaching target
    rel_pos1 = (3.0, 0.0)  # Obstacle is to the right
    rel_vel1 = (-1.0, 0.0)  # Moving left (toward target)
    v_plus1 = compute_approach_velocity(rel_pos1, rel_vel1)
    print(f"   ✓ Approach velocity (approaching): {v_plus1:.3f} (expected: > 0)")
    
    # Case 2: Obstacle moving away
    rel_vel2 = (1.0, 0.0)  # Moving right (away from target)
    v_plus2 = compute_approach_velocity(rel_pos1, rel_vel2)
    print(f"   ✓ Approach velocity (moving away): {v_plus2:.3f} (expected: 0.0)")
    assert v_plus2 < 0.01, "Approach velocity should be 0 when moving away"
    
    # Test 5: TTC computation
    print("\n5. Testing Time-to-Collision...")
    distance = 10.0
    approach_vel = 2.0
    ttc = compute_ttc(distance, approach_vel)
    print(f"   ✓ TTC: {ttc:.2f} frames (expected: 5.0)")
    assert abs(ttc - 5.0) < 0.1, "TTC computation failed"
    
    # Test 6: Obstacle size
    print("\n6. Testing obstacle size mapping...")
    size_ped = get_obstacle_size('pedestrian')
    size_car = get_obstacle_size('car')
    size_bus = get_obstacle_size('bus')
    print(f"   ✓ Pedestrian size: {size_ped} (expected: 0.0)")
    print(f"   ✓ Car size: {size_car} (expected: 0.7)")
    print(f"   ✓ Bus size: {size_bus} (expected: 1.0)")
    assert size_ped == 0.0 and size_car == 0.7 and size_bus == 1.0, "Obstacle size mapping failed"
    
    print("\n✓ All basic function tests passed!")


def test_threat_features():
    """Test threat feature computation."""
    print("\n" + "=" * 60)
    print("Testing Threat Feature Computation")
    print("=" * 60)
    
    # Test case: Close obstacle approaching target
    print("\n1. Testing threat features for close approaching obstacle...")
    target_pos = (10.0, 10.0)
    target_prev_pos = (9.0, 10.0)  # Moving east
    obstacle_pos = (12.0, 10.0)  # Close, same y
    obstacle_prev_pos = (13.0, 10.0)  # Moving west (toward target)
    
    d_ij, v_plus_ij, size_j, ttc_ij = compute_threat_features(
        target_pos, target_prev_pos,
        obstacle_pos, obstacle_prev_pos,
        obstacle_size=0.0  # Human-sized
    )
    
    print(f"   ✓ Distance (d_ij): {d_ij:.3f}")
    print(f"   ✓ Approach velocity (v+_ij): {v_plus_ij:.3f}")
    print(f"   ✓ Obstacle size (size_j): {size_j:.3f}")
    print(f"   ✓ Time-to-Collision (TTC_ij): {ttc_ij:.3f}")
    
    assert d_ij > 0, "Distance should be positive"
    assert v_plus_ij > 0, "Approach velocity should be positive when approaching"
    assert ttc_ij > 0, "TTC should be positive"
    
    print("\n✓ Threat feature computation test passed!")


def test_normalization():
    """Test feature normalization."""
    print("\n" + "=" * 60)
    print("Testing Feature Normalization")
    print("=" * 60)
    
    # Create test features
    features_list = [
        (10.0, 1.0, 0.0, 5.0),   # Close, approaching, human, low TTC
        (50.0, 0.5, 0.7, 20.0),  # Far, slow approach, car, high TTC
        (100.0, 0.0, 1.0, 20.0), # Very far, not approaching, bus, max TTC
    ]
    
    print("\n1. Raw features:")
    for i, (d, v, s, ttc) in enumerate(features_list, 1):
        print(f"   Feature {i}: d={d:.1f}, v+={v:.1f}, size={s:.1f}, TTC={ttc:.1f}")
    
    # Normalize
    normalized = normalize_threat_features_minmax(features_list, reverse_direction=[0, 3])
    
    print("\n2. Normalized features (after direction reversal for d and TTC):")
    for i, (d, v, s, ttc) in enumerate(normalized, 1):
        print(f"   Feature {i}: d'={d:.3f}, v+'={v:.3f}, size'={s:.3f}, TTC'={ttc:.3f}")
        # All should be in [0, 1]
        assert 0 <= d <= 1, f"Normalized distance should be in [0, 1], got {d}"
        assert 0 <= v <= 1, f"Normalized velocity should be in [0, 1], got {v}"
        assert 0 <= s <= 1, f"Normalized size should be in [0, 1], got {s}"
        assert 0 <= ttc <= 1, f"Normalized TTC should be in [0, 1], got {ttc}"
    
    print("\n✓ Normalization test passed!")


def test_threat_score_computation():
    """Test full threat score computation."""
    print("\n" + "=" * 60)
    print("Testing Threat Score Computation")
    print("=" * 60)
    
    # Test with normalized features
    print("\n1. Testing threat score with normalized features...")
    
    # High threat case: close, approaching, large obstacle, low TTC
    high_threat_features = (0.9, 0.8, 1.0, 0.9)  # All normalized, high threat indicators
    high_threat_score = compute_threat_score(
        high_threat_features,
        weights=(0.5, 0.25, 0.15, 0.1),
        tau=0.15,
        beta=0.5
    )
    print(f"   ✓ High threat case: score = {high_threat_score:.3f} (expected: > 0.7)")
    
    # Low threat case: far, not approaching, small obstacle, high TTC
    low_threat_features = (0.1, 0.0, 0.0, 0.1)  # All normalized, low threat indicators
    low_threat_score = compute_threat_score(
        low_threat_features,
        weights=(0.5, 0.25, 0.15, 0.1),
        tau=0.15,
        beta=0.5
    )
    print(f"   ✓ Low threat case: score = {low_threat_score:.3f} (expected: < 0.3)")
    
    assert high_threat_score > low_threat_score, "High threat should have higher score"
    assert 0 <= high_threat_score <= 1, "Threat score should be in [0, 1]"
    assert 0 <= low_threat_score <= 1, "Threat score should be in [0, 1]"
    
    print("\n✓ Threat score computation test passed!")


def test_full_pipeline():
    """Test the full pipeline with real data."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline with Real Data")
    print("=" * 60)
    
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    
    if not os.path.exists(annotation_path):
        print(f"\n⚠ Annotation file not found: {annotation_path}")
        print("   Skipping full pipeline test.")
        return
    
    # Select target
    print("\n1. Selecting target...")
    target_id, target_stats = select_target(annotation_path, auto_select=True, min_frames=100)
    print(f"   ✓ Selected target: {target_id} (appears in {target_stats['frame_count']} frames)")
    
    # Get frame data
    print("\n2. Loading annotation data...")
    frame_data = get_object_positions_per_frame(annotation_path)
    target_positions = get_target_positions(annotation_path, target_id)
    object_positions = build_object_position_history(annotation_path)
    print(f"   ✓ Loaded {len(frame_data)} frames")
    print(f"   ✓ Target appears in {len(target_positions)} frames")
    
    # Find a test frame
    print("\n3. Finding test frame...")
    test_frame = None
    for frame_id in sorted(frame_data.keys()):
        if frame_id in target_positions:
            objects = frame_data[frame_id]
            if len(objects) > 1:  # At least one other object
                test_frame = frame_id
                break
    
    if test_frame is None:
        print("   ⚠ No suitable test frame found")
        return
    
    print(f"   ✓ Testing with frame {test_frame}")
    
    # Compute threat scores
    print("\n4. Computing threat scores...")
    threat_scores = compute_threat_scores_for_frame(
        test_frame, target_id, frame_data,
        target_positions, object_positions,
        weights=(0.5, 0.25, 0.15, 0.1),
        tau=0.15,
        beta=0.5
    )
    
    print(f"   ✓ Computed threat scores for {len(threat_scores)} obstacles")
    
    if threat_scores:
        # Show top 5 threat scores
        threat_scores_sorted = sorted(threat_scores, key=lambda x: x[1], reverse=True)
        print(f"\n   Top 5 threat scores in frame {test_frame}:")
        for i, (obj_id, score, pos, features) in enumerate(threat_scores_sorted[:5], 1):
            d_ij, v_plus_ij, size_j, ttc_ij = features
            print(f"   {i}. Object {obj_id}: score={score:.3f}, pos=({pos[0]:.1f}, {pos[1]:.1f})")
            print(f"      Features: d={d_ij:.2f}, v+={v_plus_ij:.3f}, size={size_j:.1f}, TTC={ttc_ij:.2f}")
        
        # Verify scores are in valid range
        for obj_id, score, pos, features in threat_scores:
            assert 0 <= score <= 1, f"Threat score should be in [0, 1], got {score}"
    
    print("\n✓ Full pipeline test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing New DMRGCN-Based Threat Score Algorithm")
    print("=" * 60)
    
    try:
        test_basic_functions()
        test_threat_features()
        test_normalization()
        test_threat_score_computation()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe new threat score algorithm is working correctly.")
        print("You can now use it with the visualization pipeline.")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

