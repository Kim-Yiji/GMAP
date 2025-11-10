"""
Target pedestrian selection module.

This module provides utilities for selecting and validating a target pedestrian
for threat score computation.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from .data_parser import (
    load_annotations,
    organize_by_frame,
    get_available_object_ids
)


def get_object_presence_stats(annotation_path: str) -> Dict[int, Dict[str, any]]:
    """
    Get statistics about object presence across frames.
    
    Args:
        annotation_path: Path to annotation file
    
    Returns:
        Dictionary mapping object_id to statistics:
        - frame_count: Number of frames the object appears in
        - first_frame: First frame where object appears
        - last_frame: Last frame where object appears
        - frame_range: Total frames spanned
    """
    annotations = load_annotations(annotation_path)
    frame_data = organize_by_frame(annotations)
    
    # Track object presence
    object_frames = defaultdict(list)
    for frame_id, objects in frame_data.items():
        for ped_id, x, y in objects:
            object_frames[ped_id].append(frame_id)
    
    # Calculate statistics
    stats = {}
    for ped_id, frames in object_frames.items():
        frames_sorted = sorted(frames)
        stats[ped_id] = {
            'frame_count': len(frames_sorted),
            'first_frame': min(frames_sorted),
            'last_frame': max(frames_sorted),
            'frame_range': max(frames_sorted) - min(frames_sorted) + 1,
            'frames': frames_sorted
        }
    
    return stats


def get_object_average_position(annotation_path: str, video_width: Optional[float] = None, video_height: Optional[float] = None) -> Dict[int, Tuple[float, float, float]]:
    """
    Get average position for each object across all frames.
    
    Args:
        annotation_path: Path to annotation file
        video_width: Video width (optional, for center calculation)
        video_height: Video height (optional, for center calculation)
    
    Returns:
        Dictionary mapping object_id to (avg_x, avg_y, distance_from_center)
    """
    annotations = load_annotations(annotation_path)
    frame_data = organize_by_frame(annotations)
    
    # Calculate center of video if dimensions provided
    center_x = video_width / 2 if video_width else None
    center_y = video_height / 2 if video_height else None
    
    # Track positions for each object
    object_positions = defaultdict(list)
    for frame_id, objects in frame_data.items():
        for ped_id, x, y in objects:
            object_positions[ped_id].append((x, y))
    
    # Calculate average positions and distance from center
    avg_positions = {}
    for ped_id, positions in object_positions.items():
        if not positions:
            continue
        
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        
        # Calculate distance from center if center is known
        if center_x is not None and center_y is not None:
            dist_from_center = np.sqrt((avg_x - center_x)**2 + (avg_y - center_y)**2)
        else:
            dist_from_center = None
        
        avg_positions[ped_id] = (avg_x, avg_y, dist_from_center)
    
    return avg_positions


def find_central_target_candidates(
    annotation_path: str,
    video_width: float,
    video_height: float,
    min_frames: int = 100,
    max_candidates: int = 10,
    center_radius_ratio: float = 0.5,
    use_relative_center: bool = True
) -> List[Tuple[int, Dict[str, any]]]:
    """
    Find target candidates that are centrally located in the video.
    
    Args:
        annotation_path: Path to annotation file
        video_width: Video width in pixels
        video_height: Video height in pixels
        min_frames: Minimum number of frames the object should appear in
        max_candidates: Maximum number of candidates to return
        center_radius_ratio: Ratio of frame dimensions to consider as "center" (0.5 = 50% from center)
        use_relative_center: If True, use center of object cluster instead of video center
    
    Returns:
        List of tuples (object_id, stats) sorted by centrality and frame count
    """
    # Get presence stats
    stats = get_object_presence_stats(annotation_path)
    
    # Get average positions
    avg_positions = get_object_average_position(annotation_path, video_width, video_height)
    
    # Determine center point
    # Since annotations are in world coordinates (not pixel coordinates),
    # we need to find the center of the annotation coordinate space
    # This should correspond to objects that are more central in the scene
    valid_positions = [(x, y) for x, y, _ in avg_positions.values() if x is not None and y is not None]
    
    if valid_positions and use_relative_center:
        # Find objects that are in the CENTER of the annotation coordinate space
        # After coordinate transformation, objects near the center of annotation space
        # will appear near the center of the video frame
        cluster_xs = [p[0] for p in valid_positions]
        cluster_ys = [p[1] for p in valid_positions]
        
        # Find the center of the annotation coordinate RANGE (where objects are distributed)
        # This is the midpoint between min and max, which should map to video center
        min_x, max_x = min(cluster_xs), max(cluster_xs)
        min_y, max_y = min(cluster_ys), max(cluster_ys)
        center_x = (min_x + max_x) / 2  # Midpoint of X range
        center_y = (min_y + max_y) / 2  # Midpoint of Y range
        
        # Calculate acceptable radius as a fraction of the coordinate range
        range_x = max_x - min_x
        range_y = max_y - min_y
        max_range = max(range_x, range_y)
        max_radius = max_range * center_radius_ratio  # Fraction of total range
    elif valid_positions:
        # Fallback: use mean of positions (cluster center)
        cluster_xs = [p[0] for p in valid_positions]
        cluster_ys = [p[1] for p in valid_positions]
        center_x = np.mean(cluster_xs)
        center_y = np.mean(cluster_ys)
        std_x = np.std(cluster_xs)
        std_y = np.std(cluster_ys)
        max_radius = max(std_x, std_y) * 2 * center_radius_ratio
    else:
        # Final fallback: use video center (won't work well without homography)
        center_x = video_width / 2
        center_y = video_height / 2
        max_radius = min(video_width, video_height) * center_radius_ratio
    
    # Score candidates based on:
    # 1. Distance from center (closer = better)
    # 2. Frame count (more frames = better)
    candidates = []
    for ped_id, stat in stats.items():
        if stat['frame_count'] < min_frames:
            continue
        
        if ped_id not in avg_positions:
            continue
        
        avg_x, avg_y, _ = avg_positions[ped_id]
        
        if avg_x is None or avg_y is None:
            continue
        
        # Recalculate distance from the determined center
        dist_from_center = np.sqrt((avg_x - center_x)**2 + (avg_y - center_y)**2)
        
        # Only consider objects within the center radius (or top N if radius is too restrictive)
        # If max_radius is very small, include all and let scoring handle it
        if max_radius > 0 and dist_from_center > max_radius * 1.5:  # Allow 1.5x radius for flexibility
            continue
        
        # Calculate score: lower distance = higher score, more frames = higher score
        # Normalize distance (0-1, where 0 = at center, 1 = at edge of radius)
        # Use max_radius * 2 as normalization factor to give more weight to closer objects
        normalization_factor = max(max_radius * 2, 50.0)  # At least 50px
        normalized_dist = min(dist_from_center / normalization_factor, 1.0)
        centrality_score = 1.0 - normalized_dist  # Higher is better (closer to center)
        
        # Combine with frame count (weighted)
        frame_score = min(stat['frame_count'] / 5000.0, 1.0)  # Normalize frame count (up to 5000 frames)
        combined_score = centrality_score * 0.7 + frame_score * 0.3  # Weight centrality more
        
        candidates.append((
            ped_id,
            {
                **stat,
                'avg_position': (avg_x, avg_y),
                'distance_from_center': dist_from_center,
                'center_used': (center_x, center_y),
                'centrality_score': centrality_score,
                'combined_score': combined_score
            }
        ))
    
    # Sort by combined score (higher is better)
    candidates.sort(key=lambda x: x[1]['combined_score'], reverse=True)
    
    return candidates[:max_candidates]


def find_best_target_candidates(
    annotation_path: str,
    min_frames: int = 100,
    max_candidates: int = 10
) -> List[Tuple[int, Dict[str, any]]]:
    """
    Find good target pedestrian candidates based on presence statistics.
    
    Args:
        annotation_path: Path to annotation file
        min_frames: Minimum number of frames the object should appear in
        max_candidates: Maximum number of candidates to return
    
    Returns:
        List of tuples (object_id, stats) sorted by frame_count (descending)
    """
    stats = get_object_presence_stats(annotation_path)
    
    # Filter by minimum frames and sort by frame count
    candidates = [
        (ped_id, stat) for ped_id, stat in stats.items()
        if stat['frame_count'] >= min_frames
    ]
    candidates.sort(key=lambda x: x[1]['frame_count'], reverse=True)
    
    return candidates[:max_candidates]


def validate_target(
    annotation_path: str,
    target_id: int
) -> Tuple[bool, Optional[Dict[str, any]]]:
    """
    Validate that a target pedestrian exists in the annotations.
    
    Args:
        annotation_path: Path to annotation file
        target_id: Target pedestrian ID to validate
    
    Returns:
        Tuple of (is_valid, stats):
        - is_valid: True if target exists
        - stats: Dictionary with target statistics, or None if invalid
    """
    stats = get_object_presence_stats(annotation_path)
    
    if target_id not in stats:
        return False, None
    
    return True, stats[target_id]


def get_target_positions(
    annotation_path: str,
    target_id: int
) -> Dict[int, Tuple[float, float]]:
    """
    Get positions of target pedestrian across all frames.
    
    Args:
        annotation_path: Path to annotation file
        target_id: Target pedestrian ID
    
    Returns:
        Dictionary mapping frame_id to (x, y) position
    """
    annotations = load_annotations(annotation_path)
    frame_data = organize_by_frame(annotations)
    
    target_positions = {}
    for frame_id, objects in frame_data.items():
        for ped_id, x, y in objects:
            if ped_id == target_id:
                target_positions[frame_id] = (x, y)
                break  # Each frame should have at most one entry per ped_id
    
    return target_positions


def select_target(
    annotation_path: str,
    target_id: Optional[int] = None,
    auto_select: bool = False,
    min_frames: int = 100,
    video_width: Optional[float] = None,
    video_height: Optional[float] = None,
    prefer_center: bool = True
) -> Tuple[int, Dict[str, any]]:
    """
    Select a target pedestrian for threat score computation.
    
    Args:
        annotation_path: Path to annotation file
        target_id: Specific target ID to use (if provided)
        auto_select: If True and target_id is None, automatically select best candidate
        min_frames: Minimum frames for auto-selection
        video_width: Video width (optional, for center-based selection)
        video_height: Video height (optional, for center-based selection)
        prefer_center: If True and video dimensions provided, prefer centrally located targets
    
    Returns:
        Tuple of (target_id, stats)
    
    Raises:
        ValueError: If target_id is invalid or auto_select fails
    """
    if target_id is not None:
        # Validate provided target
        is_valid, stats = validate_target(annotation_path, target_id)
        if not is_valid:
            available_ids = get_available_object_ids(load_annotations(annotation_path))
            raise ValueError(
                f"Target ID {target_id} not found in annotations. "
                f"Available IDs: {available_ids[:20]}..."
            )
        return target_id, stats
    
    if auto_select:
        # Auto-select best candidate
        if prefer_center and video_width is not None and video_height is not None:
            # Prefer centrally located targets
            candidates = find_central_target_candidates(
                annotation_path,
                video_width,
                video_height,
                min_frames=min_frames,
                max_candidates=1
            )
            if candidates:
                target_id, stats = candidates[0]
                return target_id, stats
        
        # Fallback to frame count-based selection
        candidates = find_best_target_candidates(annotation_path, min_frames=min_frames, max_candidates=1)
        if not candidates:
            raise ValueError(
                f"No suitable target found with minimum {min_frames} frames. "
                f"Try reducing min_frames."
            )
        target_id, stats = candidates[0]
        return target_id, stats
    
    # Default: use first available object with reasonable presence
    candidates = find_best_target_candidates(annotation_path, min_frames=min_frames, max_candidates=1)
    if candidates:
        target_id, stats = candidates[0]
        return target_id, stats
    
    # Fallback: use any object
    available_ids = get_available_object_ids(load_annotations(annotation_path))
    if not available_ids:
        raise ValueError("No objects found in annotations")
    
    target_id = available_ids[0]
    is_valid, stats = validate_target(annotation_path, target_id)
    return target_id, stats
