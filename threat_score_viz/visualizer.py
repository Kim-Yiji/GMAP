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


def get_threat_score_color_continuous(threat_score: float, is_visible: bool = True) -> Tuple[int, int, int]:
    """
    Get color for threat score using continuous color interpolation.
    
    Uses smooth color gradient from green (low) → yellow → orange → red (high).
    This eliminates abrupt color changes by interpolating between color stops.
    
    Args:
        threat_score: Threat score in [0, 1] range
        is_visible: Whether obstacle is in field of view (if False, returns grey)
    
    Returns:
        BGR color tuple (B, G, R)
    """
    if not is_visible:
        return (200, 200, 200)  # Bright grey for non-visible obstacles
    
    # Clamp threat score to [0, 1]
    threat_score = max(0.0, min(1.0, threat_score))
    
    # Define color stops for smooth interpolation
    # Format: (threat_score, (B, G, R))
    color_stops = [
        (0.0, (0, 150, 50)),      # Dark green (low threat)
        (0.3, (0, 200, 50)),     # Bright green-yellow
        (0.5, (0, 200, 200)),    # Yellow
        (0.7, (0, 100, 255)),    # Orange
        (1.0, (0, 0, 255)),      # Bright red (high threat)
    ]
    
    # Find the two color stops to interpolate between
    if threat_score <= color_stops[0][0]:
        return color_stops[0][1]
    if threat_score >= color_stops[-1][0]:
        return color_stops[-1][1]
    
    # Find the interval
    for i in range(len(color_stops) - 1):
        score_low, color_low = color_stops[i]
        score_high, color_high = color_stops[i + 1]
        
        if score_low <= threat_score <= score_high:
            # Linear interpolation factor
            t = (threat_score - score_low) / (score_high - score_low)
            
            # Interpolate each color channel
            b = int(color_low[0] + (color_high[0] - color_low[0]) * t)
            g = int(color_low[1] + (color_high[1] - color_low[1]) * t)
            r = int(color_low[2] + (color_high[2] - color_low[2]) * t)
            
            return (b, g, r)
    
    # Fallback (shouldn't reach here)
    return (0, 255, 0)  # Green


def calculate_coordinate_transform(
    annotation_path: str,
    video_width: int,
    video_height: int,
    margin_ratio: float = 0.1,
    preserve_position: bool = False,
    manual_offset: Optional[Tuple[float, float]] = None
) -> Tuple[float, float, float, float]:
    """
    Calculate transformation parameters to map annotation coordinates to video pixel coordinates.
    
    The annotations are in a small coordinate space (e.g., 0-54), but need to be scaled
    to video pixel coordinates (e.g., 1416x1080).
    
    Args:
        annotation_path: Path to annotation file
        video_width: Video width in pixels
        video_height: Video height in pixels
        margin_ratio: Ratio of margins to leave on each side (default: 0.1 = 10%)
        preserve_position: If True, preserve upper-left position instead of centering (default: False)
        manual_offset: Optional manual (offset_x, offset_y) for calibration (default: None)
    
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
    
    if manual_offset is not None:
        # Use manually specified offset (for calibration)
        offset_x, offset_y = manual_offset
    elif preserve_position:
        # Preserve upper-left position: map (min_x, min_y) to margin position
        margin_pixels_x = video_width * margin_ratio
        margin_pixels_y = video_height * margin_ratio
        offset_x = margin_pixels_x - (min_x * scale)
        offset_y = margin_pixels_y - (min_y * scale)
    else:
        # Center the annotations in the video
        ann_center_x = (min_x + max_x) / 2
        ann_center_y = (min_y + max_y) / 2
        
        video_center_x = video_width / 2
        video_center_y = video_height / 2
        
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
    
    # Use continuous color interpolation for smooth transitions
    color = get_threat_score_color_continuous(threat_score, is_visible=True)
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
    
    # Transform coordinates if transformation is provided (with proper rounding for alignment)
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        pixel_x, pixel_y = transform_coordinates(target_position[0], target_position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        pixel_x, pixel_y = target_position[0], target_position[1]
    
    # Round to nearest integer for precise pixel alignment
    x = int(round(pixel_x))
    y = int(round(pixel_y))
    
    # Draw a smaller, less intrusive circle for the target (bright cyan/magenta)
    # Reduced sizes to not obscure the person
    # Outer black outline for contrast (smaller)
    cv2.circle(frame_copy, (x, y), 10, (0, 0, 0), 2)  # Black outline (reduced from radius 20, thickness 4)
    # Outer ring (smaller)
    cv2.circle(frame_copy, (x, y), 9, (255, 255, 0), 2)  # Cyan outer ring (reduced from radius 18, thickness 4)
    # Inner filled circle (smaller)
    cv2.circle(frame_copy, (x, y), 7, (255, 0, 255), -1)  # Magenta filled (reduced from radius 14)
    # Center dot (smaller)
    cv2.circle(frame_copy, (x, y), 3, (255, 255, 255), -1)  # White center dot (reduced from radius 6)
    
    # Draw "TARGET" label with ID (smaller font)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4 * scale  # Smaller font (reduced from 0.7)
    thickness = 1  # Thinner text (reduced from 2)
    label_text = f"TARGET ID:{target_id}"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    
    label_x = x - text_width // 2
    label_y = y - 15  # Closer to the circle (reduced from 25)
    
    # Ensure label stays within frame bounds
    frame_height, frame_width = frame.shape[:2]
    if label_x < 0:
        label_x = 5
    if label_y - text_height < 0:
        label_y = y + 15  # Adjusted for smaller marker (reduced from 30)
    
    # Draw background for label (smaller padding)
    padding = 2  # Reduced padding (from 4)
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
        1  # Thinner border (reduced from 2)
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
    alpha: float = 0.4,
    coord_transform: Optional[Tuple[float, float, float, float]] = None,
    is_visible: bool = True
) -> np.ndarray:
    """
    Draw a translucent edge (line) between target and obstacle representing threat score.
    
    Args:
        frame: Video frame (BGR format)
        target_position: (x, y) position of target in annotation coordinates
        obstacle_position: (x, y) position of obstacle in annotation coordinates
        threat_score: Threat score value [0, 1] (determines line thickness and color)
        alpha: Transparency factor (0-1, lower = more transparent, default: 0.4)
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
    
    Returns:
        Frame with translucent edge drawn
    """
    frame_copy = frame.copy()
    
    # Transform coordinates if transformation is provided (with proper rounding)
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        target_x, target_y = transform_coordinates(target_position[0], target_position[1], scale_x, scale_y, offset_x, offset_y)
        obstacle_x, obstacle_y = transform_coordinates(obstacle_position[0], obstacle_position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        target_x, target_y = target_position[0], target_position[1]
        obstacle_x, obstacle_y = obstacle_position[0], obstacle_position[1]
    
    # Round to nearest integer for precise pixel alignment
    target_x = int(round(target_x))
    target_y = int(round(target_y))
    obstacle_x = int(round(obstacle_x))
    obstacle_y = int(round(obstacle_y))
    
    # Make edges thinner: 1-3 pixels based on threat score
    thickness = max(1, int(1 + threat_score * 2))
    
    # Use continuous color interpolation for smooth transitions
    color = get_threat_score_color_continuous(threat_score, is_visible)
    
    # Create overlay for alpha blending
    overlay = frame_copy.copy()
    
    # Draw the edge line on overlay
    cv2.line(
        overlay,
        (target_x, target_y),
        (obstacle_x, obstacle_y),
        color,
        thickness,
        cv2.LINE_AA
    )
    
    # Blend overlay with original frame using alpha
    cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
    
    # Draw threat score text (smaller, optional - only for longer edges)
    distance = np.sqrt((target_x - obstacle_x)**2 + (target_y - obstacle_y)**2)
    if distance > 50:  # Only show score for longer edges to avoid clutter
        score_text = f"threat_score : {threat_score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # Smaller font
        text_thickness = 1  # Thinner text
        (text_width, text_height), baseline = cv2.getTextSize(
            score_text, font, font_scale, text_thickness
        )
        
        mid_x = (target_x + obstacle_x) // 2
        mid_y = (target_y + obstacle_y) // 2
        
        # Draw semi-transparent background for score text
        padding = 2
        overlay_text = frame_copy.copy()
        cv2.rectangle(
            overlay_text,
            (mid_x - text_width // 2 - padding, mid_y - text_height - padding),
            (mid_x + text_width // 2 + padding, mid_y + baseline + padding),
            (255, 255, 255),  # White background
            -1
        )
        cv2.addWeighted(overlay_text, 0.7, frame_copy, 0.3, 0, frame_copy)
        
        # Draw score text (opaque for readability)
        # Use blue color for score text to distinguish from ID text
        cv2.putText(
            frame_copy,
            score_text,
            (mid_x - text_width // 2, mid_y),
            font,
            font_scale,
            (255, 100, 0),  # Blue text (BGR format: B=255, G=100, R=0)
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
    coord_transform: Optional[Tuple[float, float, float, float]] = None,
    alpha: float = 0.5,
    is_visible: bool = True
) -> np.ndarray:
    """
    Draw a translucent node (circle) for an obstacle with its ID.
    
    Args:
        frame: Video frame (BGR format)
        obstacle_position: (x, y) position of obstacle in annotation coordinates
        obstacle_id: ID of the obstacle
        threat_score: Threat score value [0, 1] (determines node color and size)
        scale: Scale factor for node size
        coord_transform: Optional (scale_x, scale_y, offset_x, offset_y) transformation
        alpha: Transparency factor (0-1, lower = more transparent, default: 0.5)
    
    Returns:
        Frame with translucent obstacle node drawn
    """
    frame_copy = frame.copy()
    
    # Transform coordinates if transformation is provided (with proper rounding for alignment)
    if coord_transform:
        scale_x, scale_y, offset_x, offset_y = coord_transform
        pixel_x, pixel_y = transform_coordinates(obstacle_position[0], obstacle_position[1], scale_x, scale_y, offset_x, offset_y)
    else:
        pixel_x, pixel_y = obstacle_position[0], obstacle_position[1]
    
    # Round to nearest integer for precise pixel alignment
    x = int(round(pixel_x))
    y = int(round(pixel_y))
    
    # Use continuous color interpolation for smooth transitions (same as edge colors)
    node_color = get_threat_score_color_continuous(threat_score, is_visible)
    
    # Make nodes smaller: radius 4-8 pixels based on threat score
    node_radius = int(max(4, int(4 + threat_score * 4)) * scale)
    
    # Create overlay for alpha blending
    overlay = frame_copy.copy()
    
    # Draw translucent node circle on overlay
    cv2.circle(overlay, (x, y), node_radius, node_color, -1)  # Filled circle
    # Draw a thin outline for better visibility
    cv2.circle(overlay, (x, y), node_radius, (0, 0, 0), 1)  # Thin black outline
    
    # Blend overlay with original frame using alpha
    cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
    
    # Draw obstacle ID near the node (smaller, more subtle)
    id_text = f"id : {obstacle_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4 * scale  # Smaller font
    thickness = 1  # Thinner text
    (text_width, text_height), baseline = cv2.getTextSize(
        id_text, font, font_scale, thickness
    )
    
    # Position text above the node
    text_x = x - text_width // 2
    text_y = y - node_radius - 5
    
    # Ensure text stays within frame bounds
    frame_height, frame_width = frame.shape[:2]
    if text_x < 0:
        text_x = 5
    if text_x + text_width > frame_width:
        text_x = frame_width - text_width - 5
    if text_y - text_height < 0:
        text_y = y + node_radius + text_height + 5
    
    # Draw semi-transparent background for ID text
    overlay_text = frame_copy.copy()
    padding = 2
    cv2.rectangle(
        overlay_text,
        (text_x - padding, text_y - text_height - padding),
        (text_x + text_width + padding, text_y + baseline + padding),
        (255, 255, 255),  # White background
        -1
    )
    cv2.addWeighted(overlay_text, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Draw ID text (opaque for readability)
    # Use dark purple/magenta color for ID text to distinguish from score text
    cv2.putText(
        frame_copy,
        id_text,
        (text_x, text_y),
        font,
        font_scale,
        (128, 0, 128),  # Dark purple/magenta text (BGR format)
        thickness,
        cv2.LINE_AA
    )
    
    return frame_copy


def process_video_frame(
    frame: np.ndarray,
    frame_id: int,
    target_id: int,
    target_position: Tuple[float, float],
    threat_scores: List[Tuple[int, float, Tuple[float, float], Tuple[float, float, float, float], bool]],
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
        for obstacle_id, threat_score, position, features, is_visible in threat_scores:
            annotated_frame = draw_graph_edge(
                annotated_frame,
                target_position,
                position,
                threat_score,
                alpha=0.4,  # Semi-transparent edges
                coord_transform=coord_transform,
                is_visible=is_visible
            )
    
    # Step 2: Draw target node (most prominent)
    if draw_target:
        annotated_frame = draw_target_marker(
            annotated_frame, target_position, target_id, scale,
            coord_transform=coord_transform
        )
    
    # Step 3: Draw obstacle nodes
    if draw_graph and draw_nodes:
        for obstacle_id, threat_score, position, features, is_visible in threat_scores:
            annotated_frame = draw_obstacle_node(
                annotated_frame,
                position,
                obstacle_id,
                threat_score,
                scale,
                coord_transform=coord_transform,
                alpha=0.5,  # Semi-transparent nodes
                is_visible=is_visible
            )
    elif not draw_graph:
        # Fallback: Draw simple threat score annotations if graph is disabled
        for obstacle_id, threat_score, position, features, is_visible in threat_scores:
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
    threat_scores: List[Tuple[int, float, Tuple[float, float], Tuple[float, float, float, float], bool]]
) -> Dict:
    """
    Create metadata dictionary for a single frame.
    
    Args:
        frame_id: Frame ID
        target_id: Target pedestrian ID
        target_position: (x, y) position of target
        threat_scores: List of (obstacle_id, threat_score, position, features) tuples
            where features is (d_ij, v+_ij, size_j, TTC_ij)
    
    Returns:
        Dictionary with frame metadata
    """
    interactions = []
    for obstacle_id, threat_score, position, features, is_visible in threat_scores:
        d_ij, v_plus_ij, size_j, ttc_ij = features
        interactions.append({
            "object_id": int(obstacle_id),
            "score": float(threat_score),
            "position": [float(position[0]), float(position[1])],
            "is_visible": bool(is_visible),
            "features": {
                "d_ij_distance": float(d_ij),
                "v_plus_ij_approach_velocity": float(v_plus_ij),
                "size_j_obstacle_size": float(size_j),
                "ttc_ij_time_to_collision": float(ttc_ij)
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
    weights: Tuple[float, float, float, float] = (0.5, 0.25, 0.15, 0.1),
    tau: float = 0.15,
    beta: float = 0.5,
    eps: float = 1e-6,
    ttc_max: float = 20.0,
    obstacle_sizes: Optional[Dict[int, float]] = None,
    fov_angle: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    scale: float = 1.0,
    draw_target: bool = True,
    draw_graph: bool = True,
    draw_edges: bool = True,
    draw_nodes: bool = True,
    apply_coordinate_transform: bool = True,
    offset_x: Optional[float] = None,
    offset_y: Optional[float] = None
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
        weights: Tuple of (w_d, w_v, w_size, w_ttc) weights for threat score computation
            Default: (0.5, 0.25, 0.15, 0.1) - distance-weighted
        tau: Temperature parameter for sigmoid (controls slope, default: 0.15)
        beta: Midpoint parameter for sigmoid (controls center, default: 0.5)
        eps: Small epsilon for numerical stability (default: 1e-6)
        ttc_max: Maximum TTC value for capping (default: 20.0 frames)
        obstacle_sizes: Optional dictionary mapping object_id to size value
            If None, all obstacles are assumed to be human-sized (0.0)
        fov_angle: Optional field of view angle in radians (None = no filtering)
            e.g., np.pi/2 for 90 degrees, np.pi*100/180 for 100 degrees
        start_frame: Starting frame (default: 0)
        end_frame: Ending frame (if None, process all frames)
        scale: Scale factor for text size
        draw_target: Whether to draw target marker
        draw_graph: Whether to draw graph visualization (edges and nodes)
        draw_edges: Whether to draw edges between target and obstacles
        draw_nodes: Whether to draw nodes for obstacles
        apply_coordinate_transform: Whether to apply coordinate transformation
        offset_x: Optional manual X offset adjustment
        offset_y: Optional manual Y offset adjustment
    
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
    
    # Calculate coordinate transformation to map annotations to video pixel coordinates
    # SDD annotations are in world coordinates (small range 0-54), need transformation
    # to map to video pixel coordinates (1416x1080)
    # 
    # IMPORTANT: By default, we preserve the upper-left position of annotations
    # instead of centering them, as objects might actually be in a specific region
    # of the video (not centered). This should reduce alignment gaps.
    coord_transform = None
    if apply_coordinate_transform:
        print("Calculating coordinate transformation...")
        
        # Try to use homography file if available
        homography_path = None
        scene_name = None
        video_number = 0
        
        # Try to detect scene name from annotation path
        import os
        annotation_dir = os.path.dirname(os.path.abspath(annotation_path))
        annotation_filename = os.path.basename(annotation_path)
        
        # Common scene names in SDD
        scene_names = ['bookstore', 'deathCircle', 'gates', 'hyang', 'nexus', 'quad', 'coupa', 'little']
        for scene in scene_names:
            if scene.lower() in annotation_dir.lower() or scene.lower() in annotation_filename.lower():
                scene_name = scene.lower()
                # Try to extract video number from filename (e.g., "bookstore_video0_test.txt")
                import re
                video_match = re.search(r'video(\d+)', annotation_filename, re.IGNORECASE)
                if video_match:
                    video_number = int(video_match.group(1))
                break
        
        # Look for homography file in common locations
        homography_candidates = [
            'H_SDD.txt',
            os.path.join(os.path.dirname(annotation_path), '..', '..', 'H_SDD.txt'),
            os.path.join(os.path.dirname(annotation_path), 'H_SDD.txt'),
        ]
        for candidate in homography_candidates:
            if os.path.exists(candidate):
                homography_path = candidate
                break
        
        # Use homography if available and scene name detected
        if homography_path and scene_name:
            try:
                from .homography_parser import calculate_transform_from_homography
                from .data_parser import load_annotations
                import numpy as np
                
                print(f"  Using homography file: {homography_path}")
                print(f"  Scene: {scene_name}, Video: {video_number}")
                
                # Get annotation coordinate ranges
                annotations = load_annotations(annotation_path)
                x_coords = annotations[:, 2].astype(float)
                y_coords = annotations[:, 3].astype(float)
                
                coord_transform = calculate_transform_from_homography(
                    homography_path, scene_name, video_number,
                    width, height,
                    x_coords.min(), x_coords.max(),
                    y_coords.min(), y_coords.max()
                )
                
                scale_x, scale_y, offset_x_val, offset_y_val = coord_transform
                
                # Apply manual offset adjustments if provided
                if offset_x is not None:
                    offset_x_val += offset_x
                if offset_y is not None:
                    offset_y_val += offset_y
                
                coord_transform = (scale_x, scale_y, offset_x_val, offset_y_val)
                
                print(f"  Homography-based transformation: scale=({scale_x:.2f}, {scale_y:.2f}), offset=({offset_x_val:.2f}, {offset_y_val:.2f})")
            except Exception as e:
                print(f"  Warning: Could not use homography file: {e}")
                print("  Falling back to standard transformation...")
                homography_path = None
        
        # Fall back to standard transformation if homography not available
        if not homography_path or not scene_name:
            print("  NOTE: Using standard transformation (no homography file found)")
            print("  If objects appear misaligned, try --no-coord-transform")
            print("  to use annotations as pixel coordinates directly.")
            
            # Calculate base transformation (scale is correct since movements match)
            # Then apply offset adjustment to fix alignment
            base_transform = calculate_coordinate_transform(
                annotation_path, width, height, preserve_position=True
            )
            base_scale_x, base_scale_y, base_offset_x, base_offset_y = base_transform
            
            # Apply offset adjustment (calibrated to fix alignment gap)
            # Negative X = shift left, Positive Y = shift down
            offset_adjustment_x = offset_x if offset_x is not None else -90  # Default: shift left 90px
            offset_adjustment_y = offset_y if offset_y is not None else 100   # Default: shift down 100px
            
            adjusted_offset_x = base_offset_x + offset_adjustment_x
            adjusted_offset_y = base_offset_y + offset_adjustment_y
            
            coord_transform = (base_scale_x, base_scale_y, adjusted_offset_x, adjusted_offset_y)
            
            print(f"  Applied offset adjustment: ({offset_adjustment_x}, {offset_adjustment_y})")
            scale_x, scale_y, offset_x_val, offset_y_val = coord_transform
            print(f"  Transformation: scale=({scale_x:.2f}, {scale_y:.2f}), offset=({offset_x_val:.2f}, {offset_y_val:.2f})")
        
        # Verify transformation with target position (if using homography, coord_transform is already set)
        if coord_transform:
            scale_x, scale_y, offset_x_val, offset_y_val = coord_transform
            
            if target_id and target_id in target_positions:
                test_frame = min(target_positions.keys())
                test_pos = target_positions[test_frame]
                transformed_pos = transform_coordinates(test_pos[0], test_pos[1], scale_x, scale_y, offset_x_val, offset_y_val)
                raw_pos = (int(test_pos[0]), int(test_pos[1]))
                print(f"  Example: Target at annotation ({test_pos[0]:.1f}, {test_pos[1]:.1f})")
                print(f"    -> Transformed pixel: ({transformed_pos[0]:.1f}, {transformed_pos[1]:.1f})")
                print(f"    -> Raw pixel (no transform): {raw_pos}")
    else:
        print("Using annotation coordinates directly as pixel coordinates (no transformation)")
    
    # Determine frame range
    if end_frame is None:
        end_frame = min(video_frame_count - 1, max(target_positions.keys()))
    else:
        end_frame = min(end_frame, video_frame_count - 1, max(target_positions.keys()))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Set up video writer with better codec compatibility
    # Try H.264 first (better for web playback), fall back to mp4v
    codec_options = ['avc1', 'H264', 'mp4v']  # H.264 variants, then fallback
    fourcc = None
    out = None
    
    for codec in codec_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"  Using video codec: {codec}")
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        # Final fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"  Using video codec: mp4v (fallback)")
    
    if not out.isOpened():
        raise ValueError(f"Could not initialize video writer for: {output_video_path}")
    
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
            tau=tau,
            beta=beta,
            eps=eps,
            ttc_max=ttc_max,
            obstacle_sizes=obstacle_sizes,
            fov_angle=fov_angle
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
            "tau": float(tau),
            "beta": float(beta),
            "eps": float(eps),
            "ttc_max": float(ttc_max),
            "fov_angle": float(fov_angle) if fov_angle is not None else None,
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

