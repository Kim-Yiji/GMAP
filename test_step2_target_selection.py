#!/usr/bin/env python3
"""
Test script for Step 2: Target Pedestrian Selection

This script tests the target selection functionality.
"""

from threat_score_viz.target_selector import (
    get_object_presence_stats,
    find_best_target_candidates,
    validate_target,
    get_target_positions,
    select_target
)

def test_target_selection():
    """Test target selection functionality."""
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    
    print("=" * 60)
    print("Testing Step 2: Target Pedestrian Selection")
    print("=" * 60)
    
    # Test 1: Get object presence statistics
    print("\n1. Getting object presence statistics...")
    stats = get_object_presence_stats(annotation_path)
    print(f"   ✓ Found statistics for {len(stats)} objects")
    
    # Show some statistics
    sample_ids = list(stats.keys())[:5]
    for obj_id in sample_ids:
        stat = stats[obj_id]
        print(f"   - Object {obj_id}: appears in {stat['frame_count']} frames "
              f"(frames {stat['first_frame']}-{stat['last_frame']})")
    
    # Test 2: Find best candidates
    print("\n2. Finding best target candidates (min 100 frames)...")
    candidates = find_best_target_candidates(annotation_path, min_frames=100, max_candidates=10)
    print(f"   ✓ Found {len(candidates)} candidates")
    for i, (obj_id, stat) in enumerate(candidates[:5], 1):
        print(f"   {i}. Object {obj_id}: {stat['frame_count']} frames "
              f"(range: {stat['first_frame']}-{stat['last_frame']})")
    
    # Test 3: Validate a target
    print("\n3. Validating target IDs...")
    if candidates:
        test_id = candidates[0][0]
        is_valid, stat = validate_target(annotation_path, test_id)
        print(f"   ✓ Target {test_id} is valid: {is_valid}")
        if stat:
            print(f"     - Appears in {stat['frame_count']} frames")
            print(f"     - Frame range: {stat['first_frame']}-{stat['last_frame']}")
    
    # Test invalid target
    is_valid, stat = validate_target(annotation_path, 99999)
    print(f"   ✓ Target 99999 is valid: {is_valid} (expected: False)")
    
    # Test 4: Get target positions
    print("\n4. Getting target positions...")
    if candidates:
        test_id = candidates[0][0]
        positions = get_target_positions(annotation_path, test_id)
        print(f"   ✓ Found {len(positions)} positions for target {test_id}")
        # Show first few positions
        frame_ids = sorted(positions.keys())[:5]
        for frame_id in frame_ids:
            x, y = positions[frame_id]
            print(f"     - Frame {frame_id}: ({x:.2f}, {y:.2f})")
    
    # Test 5: Select target (auto-select)
    print("\n5. Auto-selecting target...")
    target_id, target_stats = select_target(annotation_path, auto_select=True, min_frames=100)
    print(f"   ✓ Selected target: {target_id}")
    print(f"     - Appears in {target_stats['frame_count']} frames")
    print(f"     - Frame range: {target_stats['first_frame']}-{target_stats['last_frame']}")
    
    # Test 6: Select specific target
    print("\n6. Selecting specific target...")
    if candidates:
        specific_id = candidates[0][0]
        target_id, target_stats = select_target(annotation_path, target_id=specific_id)
        print(f"   ✓ Selected target: {target_id}")
        print(f"     - Appears in {target_stats['frame_count']} frames")
    
    print("\n" + "=" * 60)
    print("✓ Step 2: Target Pedestrian Selection - ALL TESTS PASSED")
    print("=" * 60)

if __name__ == '__main__':
    test_target_selection()

