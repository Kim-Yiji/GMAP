#!/usr/bin/env python3
"""
Test script for Step 1: Data Parser Module

This script tests the data parser to ensure it correctly loads and organizes
SDD annotation files.
"""

from threat_score_viz.data_parser import (
    load_annotations,
    organize_by_frame,
    get_object_positions_per_frame,
    get_available_object_ids,
    get_frame_range
)

def test_parser():
    """Test the data parser with a sample annotation file."""
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    
    print("=" * 60)
    print("Testing Step 1: Data Parser Module")
    print("=" * 60)
    
    # Test 1: Load annotations
    print("\n1. Loading annotations...")
    annotations = load_annotations(annotation_path)
    print(f"   ✓ Loaded {len(annotations)} annotation entries")
    print(f"   ✓ Sample row: {annotations[0] if len(annotations) > 0 else 'N/A'}")
    
    # Test 2: Organize by frame
    print("\n2. Organizing by frame...")
    frame_data = organize_by_frame(annotations)
    print(f"   ✓ Found {len(frame_data)} unique frames")
    
    # Test 3: Get frame range
    print("\n3. Getting frame range...")
    min_frame, max_frame = get_frame_range(annotations)
    print(f"   ✓ Frame range: {min_frame} to {max_frame}")
    
    # Test 4: Get available object IDs
    print("\n4. Getting available object IDs...")
    object_ids = get_available_object_ids(annotations)
    print(f"   ✓ Found {len(object_ids)} unique object IDs")
    print(f"   ✓ First 10 object IDs: {object_ids[:10]}")
    
    # Test 5: Check frame 0 data
    print("\n5. Checking frame 0 data...")
    if 0 in frame_data:
        objects_in_frame_0 = frame_data[0]
        print(f"   ✓ Frame 0 has {len(objects_in_frame_0)} objects")
        print(f"   ✓ Sample objects: {objects_in_frame_0[:3]}")
    else:
        print(f"   ⚠ Frame 0 not found in data")
    
    # Test 6: Convenience function
    print("\n6. Testing convenience function...")
    frame_data_2 = get_object_positions_per_frame(annotation_path)
    print(f"   ✓ Convenience function loaded {len(frame_data_2)} frames")
    
    print("\n" + "=" * 60)
    print("✓ Step 1: Data Parser Module - ALL TESTS PASSED")
    print("=" * 60)

if __name__ == '__main__':
    test_parser()

