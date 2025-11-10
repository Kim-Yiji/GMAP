"""
Video utility functions for loading and verifying video files.

This module provides functions to:
- Load video files and get their properties
- Verify that annotation frame counts match video frame counts
"""

import cv2
import os
from typing import Tuple, Optional, Dict


def get_video_properties(video_path: str) -> Dict[str, any]:
    """
    Get video properties (frame count, FPS, resolution, etc.).
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with video properties:
        - frame_count: Total number of frames
        - fps: Frames per second
        - width: Video width in pixels
        - height: Video height in pixels
        - duration: Video duration in seconds
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }


def verify_annotation_video_alignment(
    annotation_path: str,
    video_path: str,
    annotation_frame_range: Optional[Tuple[int, int]] = None
) -> Tuple[bool, Dict[str, any]]:
    """
    Verify that annotation frame IDs align with video frame count.
    
    Args:
        annotation_path: Path to annotation file
        video_path: Path to video file
        annotation_frame_range: Optional tuple (min_frame, max_frame) from annotations.
                               If None, will be computed from annotation file.
    
    Returns:
        Tuple of (is_aligned, info_dict):
        - is_aligned: True if frame counts match
        - info_dict: Dictionary with alignment information
    """
    from .data_parser import load_annotations, get_frame_range
    
    # Get video properties
    video_props = get_video_properties(video_path)
    video_frame_count = video_props['frame_count']
    
    # Get annotation frame range
    if annotation_frame_range is None:
        annotations = load_annotations(annotation_path)
        min_frame, max_frame = get_frame_range(annotations)
        annotation_frame_count = max_frame - min_frame + 1
    else:
        min_frame, max_frame = annotation_frame_range
        annotation_frame_count = max_frame - min_frame + 1
    
    # Check alignment
    is_aligned = (
        min_frame == 0 and
        max_frame == video_frame_count - 1 and
        annotation_frame_count == video_frame_count
    )
    
    info = {
        'video_frame_count': video_frame_count,
        'annotation_min_frame': min_frame,
        'annotation_max_frame': max_frame,
        'annotation_frame_count': annotation_frame_count,
        'is_aligned': is_aligned,
        'frame_count_match': annotation_frame_count == video_frame_count,
        'starts_at_zero': min_frame == 0,
        'ends_at_video_end': max_frame == video_frame_count - 1,
        'video_properties': video_props
    }
    
    return is_aligned, info


def load_video_frame(video_path: str, frame_id: int) -> Optional[Tuple[bool, any]]:
    """
    Load a specific frame from a video.
    
    Args:
        video_path: Path to the video file
        frame_id: Frame number to load (0-indexed)
    
    Returns:
        Tuple of (success, frame) where:
        - success: True if frame was loaded successfully
        - frame: numpy array of the frame (BGR format), or None if failed
    """
    if not os.path.exists(video_path):
        return False, None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False, None
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    # Read frame
    ret, frame = cap.read()
    
    cap.release()
    
    if ret:
        return True, frame
    else:
        return False, None

