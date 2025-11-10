#!/usr/bin/env python3
"""
Test script to verify video-annotation alignment.

This script checks that annotation frame IDs match the video frame count.
"""

from threat_score_viz.video_utils import verify_annotation_video_alignment, get_video_properties
from threat_score_viz.data_parser import get_frame_range, load_annotations

def test_alignment():
    """Test video-annotation alignment for bookstore_video0."""
    annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
    video_path = 'sdd_bookstore/bookstore_vid/video0/video.mp4'
    
    print("=" * 60)
    print("Video-Annotation Alignment Verification")
    print("=" * 60)
    
    # Get video properties
    print("\n1. Video Properties:")
    video_props = get_video_properties(video_path)
    print(f"   ✓ Frame count: {video_props['frame_count']}")
    print(f"   ✓ FPS: {video_props['fps']:.2f}")
    print(f"   ✓ Resolution: {video_props['width']}x{video_props['height']}")
    print(f"   ✓ Duration: {video_props['duration']:.2f} seconds")
    
    # Verify alignment
    print("\n2. Alignment Verification:")
    is_aligned, info = verify_annotation_video_alignment(annotation_path, video_path)
    
    if is_aligned:
        print("   ✓ ALIGNED: Annotation frames match video frames perfectly")
    else:
        print("   ✗ MISALIGNED: There is a mismatch between annotation and video")
    
    print(f"\n   Annotation details:")
    print(f"   - Frame range: {info['annotation_min_frame']} to {info['annotation_max_frame']}")
    print(f"   - Total frames: {info['annotation_frame_count']}")
    print(f"   - Video frames: {info['video_frame_count']}")
    print(f"   - Frame count match: {info['frame_count_match']}")
    print(f"   - Starts at 0: {info['starts_at_zero']}")
    print(f"   - Ends at video end: {info['ends_at_video_end']}")
    
    print("\n" + "=" * 60)
    if is_aligned:
        print("✓ VERIFICATION PASSED: Ready to proceed with visualization")
    else:
        print("⚠ WARNING: Frame count mismatch detected")
    print("=" * 60)

if __name__ == '__main__':
    test_alignment()

