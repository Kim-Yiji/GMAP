"""
Visualization module for threat scores.

This module provides functions to:
- Overlay threat scores on video frames
- Draw text annotations with object IDs and scores
- Save annotated videos
- Generate metadata for visualization frameworks
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .data_parser import get_object_positions_per_frame
from .target_selector import get_target_positions, select_target
from .threat_score_computer import (
    compute_threat_scores_for_frame,
    build_object_position_history
)
from .video_utils import get_video_properties, load_video_frame


def calculate_coordinate_transform(
    annotation_path: str,
    video_width: int,
    video_height: int,
    margin_ratio: float = 0.1
) -> Tuple[float, float, float, float]:
    """
    Calculate transformation parameters to map annotation coordinates to video pixel coordinates.
    
    The annotations are in a small coordinate space (e.g., 0-54), but need to be scaled
    and centered in the video frame (e.g., 1416x1080).
    
    Args:
        annotation_path: Path to annotation file
        video_width: Video width in pixels
        video_height: Video height in pixels
        margin_ratio: Ratio of margins to leave on each side (default: 0.1 = 10%)
    
    Returns:
        Tuple of (scale_x, scale_y, offset_x, offset_y) transformation parameters
    """
    # Load all annotations to find coordinate ranges
    from .data_parser import load_annotations
    annotations = load_annotations(annotation_path)
    
    if len(annotations) == 0:
        # Default: no transformation
        return (1.0, 1.0, 0.0, 0.0)
    
    # Get coordinate ranges
    xs = annotations[:, 2]
    ys = annotations[:, 3]
    
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    
    # Calculate annotation coordinate ranges
    ann_range_x = max_x - min_x
    ann_range_y = max_y - min_y
    
    if ann_range_x == 0 or ann_range_y == 0:
        # No range, use default
        return (1.0, 1.0, 0.0, 0.0)
    
    # Calculate available video space (with margins)
    available_width = video_width * (1 - 2 * margin_ratio)
    available_height = video_height * (1 - 2 * margin_ratio)
    
    # Calculate scaling factors to fit annotation range into available space
    scale_x = available_width / ann_range_x
    scale_y = available_height / ann_range_y
    
    # Use uniform scaling (maintain aspect ratio) to avoid distortion
    scale = min(scale_x, scale_y)
    
    # Calculate center of annotation coordinates
    ann_center_x = (min_x + max_x) / 2
    ann_center_y = (min_y + max_y) / 2
    
    # Calculate center of video (with margins)
    video_center_x = video_width / 2
    video_center_y = video_height / 2
    
    # Calculate offsets to center the annotations
    offset_x = video_center_x - (ann_center_x * scale)
    offset_y = video_center_y - (ann_center_y * scale)
    
    return (scale, scale, offset_x, offset_y)


def transform_coordinates(
    x: float,
    y: float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float
) -> Tuple[float, float]:
    """
    Transform annotation coordinates to video pixel coordinates.
    
    Args:
        x: Annotation x coordinate
        y: Annotation y coordinate
        scale_x: X scaling factor
        scale_y: Y scaling factor
        offset_x: X offset
        offset_y: Y offset
    
    Returns:
        Tuple of (pixel_x, pixel_y)
    """
    pixel_x = x * scale_x + offset_x
    pixel_y = y * scale_y + offset_y
    return (pixel_x, pixel_y)


def draw_threat_score_on_frame(
    frame: np.ndarray,
    obstacle_id: int,
    threat_score: float,
    position: Tuple[float, float],
    target_position: Tuple[float, float],
    scale: float = 1.0,
    coord_transform: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Draw threat score text on a video frame.
    
    Args:
        frame: Video frame (BGR format)
        obstacle_id: ID of the obstacle
        threat_score: Threat score value [0, 1]
        position: (x, y) position of the obstacle in annotation coordinates
        target_position: (x, y) position of the target (for reference)
        scale: Scale factor for text size (default: 1.0)
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
    
    Returns:
        Frame with threat score annotation drawn
    """
    frame_copy = frame.copy()
    
    # Transform coordinates if transformation is provided
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        pixel_x, pixel_y = transform_coordinates(position[0], position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        pixel_x, pixel_y = position[0], position[1]
    
    # Convert position to integers (image coordinates)
    x = int(pixel_x)
    y = int(pixel_y)
    
    # Prepare text
    score_text = f"ID:{obstacle_id} {threat_score:.2f}"
    
    # Choose text color based on threat score (red = high threat, green = low threat)
    # High threat (>= 0.7): Red
    # Medium threat (0.4-0.7): Yellow/Orange
    # Low threat (< 0.4): Green
    if threat_score >= 0.7:
        color = (0, 0, 255)  # Red in BGR
        bg_color = (0, 0, 0)  # Black background
    elif threat_score >= 0.4:
        color = (0, 165, 255)  # Orange in BGR
        bg_color = (0, 0, 0)  # Black background
    else:
        color = (0, 255, 0)  # Green in BGR
        bg_color = (0, 0, 0)  # Black background
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        score_text, font, font_scale, thickness
    )
    
    # Offset text position (above the object, slightly to the right)
    text_x = x + 5
    text_y = y - 10
    
    # Ensure text stays within frame bounds
    frame_height, frame_width = frame.shape[:2]
    if text_x + text_width > frame_width:
        text_x = x - text_width - 5
    if text_y - text_height < 0:
        text_y = y + text_height + 10
    
    # Draw background rectangle for better text visibility
    cv2.rectangle(
        frame_copy,
        (text_x - 2, text_y - text_height - 2),
        (text_x + text_width + 2, text_y + baseline + 2),
        bg_color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        frame_copy,
        score_text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    # Optionally draw a small circle at the obstacle position
    cv2.circle(frame_copy, (x, y), 3, color, -1)
    
    return frame_copy


def draw_target_marker(
    frame: np.ndarray,
    target_position: Tuple[float, float],
    target_id: int,
    scale: float = 1.0,
    coord_transform: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Draw a prominent marker for the target pedestrian.
    
    Args:
        frame: Video frame (BGR format)
        target_position: (x, y) position of the target in annotation coordinates
        target_id: ID of the target pedestrian
        scale: Scale factor for marker size
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
    
    Returns:
        Frame with target marker drawn
    """
    frame_copy = frame.copy()
    
    # Transform coordinates if transformation is provided
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        pixel_x, pixel_y = transform_coordinates(target_position[0], target_position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        pixel_x, pixel_y = target_position[0], target_position[1]
    
    x = int(pixel_x)
    y = int(pixel_y)
    
    # Draw a larger, more prominent circle for the target (bright cyan/magenta)
    # Outer black outline for contrast
    cv2.circle(frame_copy, (x, y), 20, (0, 0, 0), 4)  # Black outline
    # Outer ring
    cv2.circle(frame_copy, (x, y), 18, (255, 255, 0), 4)  # Cyan outer ring (thicker)
    # Inner filled circle
    cv2.circle(frame_copy, (x, y), 14, (255, 0, 255), -1)  # Magenta filled (larger)
    # Center dot
    cv2.circle(frame_copy, (x, y), 6, (255, 255, 255), -1)  # White center dot
    
    # Draw "TARGET" label with ID
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 * scale
    thickness = 2
    label_text = f"TARGET ID:{target_id}"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    
    label_x = x - text_width // 2
    label_y = y - 25
    
    # Ensure label stays within frame bounds
    frame_height, frame_width = frame.shape[:2]
    if label_x < 0:
        label_x = 5
    if label_y - text_height < 0:
        label_y = y + 30
    
    # Draw background for label (semi-transparent effect with multiple rectangles)
    padding = 4
    cv2.rectangle(
        frame_copy,
        (label_x - padding, label_y - text_height - padding),
        (label_x + text_width + padding, label_y + baseline + padding),
        (0, 0, 0),
        -1
    )
    cv2.rectangle(
        frame_copy,
        (label_x - padding - 1, label_y - text_height - padding - 1),
        (label_x + text_width + padding + 1, label_y + baseline + padding + 1),
        (255, 255, 0),  # Cyan border
        2
    )
    
    # Draw label
    cv2.putText(
        frame_copy,
        label_text,
        (label_x, label_y),
        font,
        font_scale,
        (255, 255, 0),  # Cyan text
        thickness,
        cv2.LINE_AA
    )
    
    return frame_copy


def draw_graph_edge(
    frame: np.ndarray,
    target_position: Tuple[float, float],
    obstacle_position: Tuple[float, float],
    threat_score: float,
    alpha: float = 0.6,
    coord_transform: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Draw an edge (line) between target and obstacle representing threat score.
    
    Args:
        frame: Video frame (BGR format)
        target_position: (x, y) position of target in annotation coordinates
        obstacle_position: (x, y) position of obstacle in annotation coordinates
        threat_score: Threat score value [0, 1] (determines line thickness and color)
        alpha: Transparency factor (0-1, higher = more opaque)
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
    
    Returns:
        Frame with edge drawn
    """
    frame_copy = frame.copy()
    
    # Transform coordinates if transformation is provided
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        target_x, target_y = transform_coordinates(target_position[0], target_position[1], scale_x, scale_y, offset_x, offset_y)
        obstacle_x, obstacle_y = transform_coordinates(obstacle_position[0], obstacle_position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        target_x, target_y = target_position[0], target_position[1]
        obstacle_x, obstacle_y = obstacle_position[0], obstacle_position[1]
    
    target_x = int(target_x)
    target_y = int(target_y)
    obstacle_x = int(obstacle_x)
    obstacle_y = int(obstacle_y)
    
    # Determine line thickness based on threat score (2-8 pixels for better visibility)
    # Higher threat = thicker line
    thickness = max(2, int(2 + threat_score * 6))
    
    # Determine line color based on threat score (brighter colors for visibility)
    # High threat (>= 0.7): Bright Red
    # Medium threat (0.4-0.7): Orange/Yellow
    # Low threat (< 0.4): Green
    if threat_score >= 0.7:
        color = (0, 0, 255)  # Bright Red in BGR
    elif threat_score >= 0.4:
        # Interpolate between yellow and red
        ratio = (threat_score - 0.4) / 0.3
        color = (0, int(255 * (1 - ratio)), 255)  # Yellow to Orange
    else:
        # Interpolate between green and yellow
        ratio = threat_score / 0.4
        color = (0, 255, int(255 * ratio))  # Green to Yellow
    
    # Draw a thicker outline first for better visibility
    outline_thickness = thickness + 2
    cv2.line(
        frame_copy,
        (target_x, target_y),
        (obstacle_x, obstacle_y),
        (0, 0, 0),  # Black outline
        outline_thickness,
        cv2.LINE_AA
    )
    
    # Draw the main edge line
    cv2.line(
        frame_copy,
        (target_x, target_y),
        (obstacle_x, obstacle_y),
        color,
        thickness,
        cv2.LINE_AA
    )
    
    # Draw threat score along the edge (midpoint)
    mid_x = (target_x + obstacle_x) // 2
    mid_y = (target_y + obstacle_y) // 2
    
    # Draw score text if line is long enough (lowered threshold for visibility)
    distance = np.sqrt((target_x - obstacle_x)**2 + (target_y - obstacle_y)**2)
    if distance > 20:  # Lowered threshold to show more scores
        score_text = f"{threat_score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Slightly larger
        text_thickness = 2  # Thicker text
        (text_width, text_height), baseline = cv2.getTextSize(
            score_text, font, font_scale, text_thickness
        )
        
        # Draw background for score text (larger padding)
        padding = 4
        cv2.rectangle(
            frame_copy,
            (mid_x - text_width // 2 - padding, mid_y - text_height - padding),
            (mid_x + text_width // 2 + padding, mid_y + baseline + padding),
            (255, 255, 255),  # White background for better contrast
            -1
        )
        cv2.rectangle(
            frame_copy,
            (mid_x - text_width // 2 - padding, mid_y - text_height - padding),
            (mid_x + text_width // 2 + padding, mid_y + baseline + padding),
            (0, 0, 0),  # Black border
            2
        )
        
        # Draw score text
        cv2.putText(
            frame_copy,
            score_text,
            (mid_x - text_width // 2, mid_y),
            font,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA
        )
    
    return frame_copy


def draw_obstacle_node(
    frame: np.ndarray,
    obstacle_position: Tuple[float, float],
    obstacle_id: int,
    threat_score: float,
    scale: float = 1.0,
    coord_transform: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Draw a node (circle) for an obstacle with its ID.
    
    Args:
        frame: Video frame (BGR format)
        obstacle_position: (x, y) position of obstacle in annotation coordinates
        obstacle_id: ID of the obstacle
        threat_score: Threat score value [0, 1] (determines node color)
        scale: Scale factor for node size
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
    
    Returns:
        Frame with obstacle node drawn
    """
    frame_copy = frame.copy()
    
    # Transform coordinates if transformation is provided
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        pixel_x, pixel_y = transform_coordinates(obstacle_position[0], obstacle_position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        pixel_x, pixel_y = obstacle_position[0], obstacle_position[1]
    
    x = int(pixel_x)
    y = int(pixel_y)
    
    # Determine node color based on threat score (same as edge colors)
    if threat_score >= 0.7:
        node_color = (0, 0, 255)  # Red
    elif threat_score >= 0.4:
        ratio = (threat_score - 0.4) / 0.3
        node_color = (0, int(255 * (1 - ratio)), 255)  # Yellow to Orange
    else:
        ratio = threat_score / 0.4
        node_color = (0, 255, int(255 * ratio))  # Green to Yellow
    
    # Draw node circle (size based on threat score, larger for better visibility)
    node_radius = int(max(8, int(8 + threat_score * 8)) * scale)
    # Draw outline for better visibility
    cv2.circle(frame_copy, (x, y), node_radius + 2, (0, 0, 0), 2)  # Black outline
    cv2.circle(frame_copy, (x, y), node_radius, node_color, 3)  # Thicker border
    inner_radius = max(3, node_radius - 3)
    cv2.circle(frame_copy, (x, y), inner_radius, node_color, -1)  # Filled center
    
    # Draw obstacle ID near the node (more prominent)
    id_text = f"ID:{obstacle_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale  # Larger font
    thickness = 2  # Thicker text
    (text_width, text_height), baseline = cv2.getTextSize(
        id_text, font, font_scale, thickness
    )
    
    # Position text above or below the node
    text_x = x - text_width // 2
    text_y = y - node_radius - 10
    
    # Ensure text stays within frame bounds
    frame_height, frame_width = frame.shape[:2]
    if text_x < 0:
        text_x = 5
    if text_x + text_width > frame_width:
        text_x = frame_width - text_width - 5
    if text_y - text_height < 0:
        text_y = y + node_radius + text_height + 10
    
    # Draw background for ID text (white with black border for contrast)
    padding = 3
    cv2.rectangle(
        frame_copy,
        (text_x - padding, text_y - text_height - padding),
        (text_x + text_width + padding, text_y + baseline + padding),
        (255, 255, 255),  # White background
        -1
    )
    cv2.rectangle(
        frame_copy,
        (text_x - padding, text_y - text_height - padding),
        (text_x + text_width + padding, text_y + baseline + padding),
        (0, 0, 0),  # Black border
        1
    )
    
    # Draw ID text
    cv2.putText(
        frame_copy,
        id_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),  # Black text for contrast
        thickness,
        cv2.LINE_AA
    )
    
    return frame_copy


def process_video_frame(
    frame: np.ndarray,
    frame_id: int,
    target_id: int,
    target_position: Tuple[float, float],
    threat_scores: List[Tuple[int, float, Tuple[float, float], Tuple[float, float, float, float]]],
    draw_target: bool = True,
    draw_graph: bool = True,
    draw_edges: bool = True,
    draw_nodes: bool = True,
    scale: float = 1.0,
    coord_transform: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Process a single video frame by drawing graph visualization with threat scores.
    
    Args:
        frame: Video frame (BGR format)
        frame_id: Frame ID
        target_id: Target pedestrian ID
        target_position: (x, y) position of target in annotation coordinates
        threat_scores: List of (obstacle_id, threat_score, position, features) tuples
        draw_target: Whether to draw target marker
        draw_graph: Whether to draw graph visualization (edges and nodes)
        draw_edges: Whether to draw edges between target and obstacles
        draw_nodes: Whether to draw nodes for obstacles
        scale: Scale factor for text size
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
    
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    # Step 1: Draw graph edges first (so nodes appear on top)
    if draw_graph and draw_edges:
        for obstacle_id, threat_score, position, features in threat_scores:
            annotated_frame = draw_graph_edge(
                annotated_frame,
                target_position,
                position,
                threat_score,
                coord_transform=coord_transform
            )
    
    # Step 2: Draw target node (most prominent)
    if draw_target:
        annotated_frame = draw_target_marker(
            annotated_frame, target_position, target_id, scale,
            coord_transform=coord_transform
        )
    
    # Step 3: Draw obstacle nodes
    if draw_graph and draw_nodes:
        for obstacle_id, threat_score, position, features in threat_scores:
            annotated_frame = draw_obstacle_node(
                annotated_frame,
                position,
                obstacle_id,
                threat_score,
                scale,
                coord_transform=coord_transform
            )
    elif not draw_graph:
        # Fallback: Draw simple threat score annotations if graph is disabled
        for obstacle_id, threat_score, position, features in threat_scores:
            annotated_frame = draw_threat_score_on_frame(
                annotated_frame,
                obstacle_id,
                threat_score,
                position,
                target_position,
                scale,
                coord_transform=coord_transform
            )
    
    return annotated_frame


def create_frame_metadata(
    frame_id: int,
    target_id: int,
    target_position: Tuple[float, float],
    threat_scores: List[Tuple[int, float, Tuple[float, float], Tuple[float, float, float, float]]]
) -> Dict:
    """
    Create metadata dictionary for a single frame.
    
    Args:
        frame_id: Frame ID
        target_id: Target pedestrian ID
        target_position: (x, y) position of target
        threat_scores: List of (obstacle_id, threat_score, position, features) tuples
    
    Returns:
        Dictionary with frame metadata
    """
    interactions = []
    for obstacle_id, threat_score, position, features in threat_scores:
        f1, f2, f3, f4 = features
        interactions.append({
            "object_id": int(obstacle_id),
            "score": float(threat_score),
            "position": [float(position[0]), float(position[1])],
            "features": {
                "f1_distance": float(f1),
                "f2_velocity_diff": float(f2),
                "f3_heading_alignment": float(f3),
                "f4_class_interaction": float(f4)
            }
        })
    
    return {
        "frame_id": int(frame_id),
        "target_id": int(target_id),
        "target_position": [float(target_position[0]), float(target_position[1])],
        "interactions": interactions
    }


def process_video_with_threat_scores(
    video_path: str,
    annotation_path: str,
    output_video_path: str,
    output_metadata_path: str,
    target_id: Optional[int] = None,
    auto_select_target: bool = True,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    max_distance: float = 100.0,
    max_vel_diff: float = 10.0,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    scale: float = 1.0,
    draw_target: bool = True,
    draw_graph: bool = True,
    draw_edges: bool = True,
    draw_nodes: bool = True
) -> Dict:
    """
    Process video and overlay threat scores on each frame with graph visualization.
    
    Args:
        video_path: Path to input video file
        annotation_path: Path to annotation file
        output_video_path: Path to save annotated video
        output_metadata_path: Path to save metadata JSON file
        target_id: Target pedestrian ID (if None, will auto-select)
        auto_select_target: Whether to auto-select target if target_id is None
        weights: Tuple of (w1, w2, w3, w4) weights for threat score computation
        max_distance: Maximum distance for normalization
        max_vel_diff: Maximum velocity difference for normalization
        start_frame: Starting frame (default: 0)
        end_frame: Ending frame (if None, process all frames)
        scale: Scale factor for text size
        draw_target: Whether to draw target marker
        draw_graph: Whether to draw graph visualization (edges and nodes)
        draw_edges: Whether to draw edges between target and obstacles
        draw_nodes: Whether to draw nodes for obstacles
    
    Returns:
        Dictionary with processing statistics
    """
    # Get video properties first (needed for center-based selection)
    video_props = get_video_properties(video_path)
    video_frame_count = video_props['frame_count']
    fps = video_props['fps']
    width = video_props['width']
    height = video_props['height']
    
    # Load annotations and select target
    print("Loading annotations...")
    frame_data = get_object_positions_per_frame(annotation_path)
    
    if target_id is None:
        if auto_select_target:
            # Prefer centrally located targets
            target_id, target_stats = select_target(
                annotation_path, 
                auto_select=True,
                video_width=width,
                video_height=height,
                prefer_center=True
            )
            if 'avg_position' in target_stats:
                avg_x, avg_y = target_stats['avg_position']
                dist = target_stats.get('distance_from_center', 'N/A')
                print(f"Auto-selected central target: {target_id} (appears in {target_stats['frame_count']} frames)")
                print(f"  Average position: ({avg_x:.1f}, {avg_y:.1f}), Distance from center: {dist:.1f}px")
            else:
                print(f"Auto-selected target: {target_id} (appears in {target_stats['frame_count']} frames)")
        else:
            raise ValueError("target_id must be provided if auto_select_target is False")
    else:
        from .target_selector import validate_target
        is_valid, target_stats = validate_target(annotation_path, target_id)
        if not is_valid:
            raise ValueError(f"Invalid target_id: {target_id}")
        print(f"Using target: {target_id} (appears in {target_stats['frame_count']} frames)")
    
    # Get target positions and object position history
    target_positions = get_target_positions(annotation_path, target_id)
    object_positions = build_object_position_history(annotation_path)
    
    # Calculate coordinate transformation to center annotations in video
    print("Calculating coordinate transformation...")
    coord_transform = calculate_coordinate_transform(annotation_path, width, height)
    scale_x, scale_y, offset_x, offset_y = coord_transform
    print(f"  Scale: ({scale_x:.2f}, {scale_y:.2f}), Offset: ({offset_x:.2f}, {offset_y:.2f})")
    
    # Determine frame range
    if end_frame is None:
        end_frame = min(video_frame_count - 1, max(target_positions.keys()))
    else:
        end_frame = min(end_frame, video_frame_count - 1, max(target_positions.keys()))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process frames
    metadata_list = []
    frames_processed = 0
    frames_skipped = 0
    
    print(f"Processing frames {start_frame} to {end_frame}...")
    for frame_id in tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if not ret:
            frames_skipped += 1
            continue
        
        # Check if target exists in this frame
        if frame_id not in target_positions:
            frames_skipped += 1
            # Still write the frame (without annotations)
            out.write(frame)
            continue
        
        target_position = target_positions[frame_id]
        
        # Compute threat scores for this frame
        threat_scores = compute_threat_scores_for_frame(
            frame_id, target_id, frame_data,
            target_positions, object_positions,
            weights=weights,
            max_distance=max_distance,
            max_vel_diff=max_vel_diff
        )
        
        # Process frame with annotations
        annotated_frame = process_video_frame(
            frame, frame_id, target_id, target_position,
            threat_scores, 
            draw_target=draw_target, 
            draw_graph=draw_graph,
            draw_edges=draw_edges,
            draw_nodes=draw_nodes,
            scale=scale,
            coord_transform=coord_transform
        )
        
        # Write annotated frame
        out.write(annotated_frame)
        
        # Create and store metadata
        frame_metadata = create_frame_metadata(
            frame_id, target_id, target_position, threat_scores
        )
        metadata_list.append(frame_metadata)
        
        frames_processed += 1
    
    # Clean up
    cap.release()
    out.release()
    
    # Save metadata
    metadata = {
        "video_path": video_path,
        "annotation_path": annotation_path,
        "target_id": int(target_id),
        "video_properties": {
            "frame_count": int(video_frame_count),
            "fps": float(fps),
            "width": int(width),
            "height": int(height)
        },
        "processing_settings": {
            "weights": [float(w) for w in weights],
            "max_distance": float(max_distance),
            "max_vel_diff": float(max_vel_diff),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame)
        },
        "frames": metadata_list
    }
    
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Return statistics
    stats = {
        "frames_processed": frames_processed,
        "frames_skipped": frames_skipped,
        "total_frames": end_frame - start_frame + 1,
        "output_video": output_video_path,
        "output_metadata": output_metadata_path
    }
    
    return stats

