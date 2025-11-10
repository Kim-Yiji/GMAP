"""
Threat score computation module.

This module computes threat scores based on 4 relational features:
- f1: Distance between target and obstacle
- f2: Velocity difference (relative speed)
- f3: Heading alignment (direction similarity)
- f4: Object class interaction (placeholder)

Threat score = sigmoid(w1*f1 + w2*f2 + w3*f3 + w4*f4)
where w1 = w2 = w3 = w4 = 1 (uniform weights)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .data_parser import get_object_positions_per_frame
from .target_selector import get_target_positions


def compute_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two positions.
    
    Args:
        pos1: (x, y) position of first object
        pos2: (x, y) position of second object
    
    Returns:
        Euclidean distance
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def compute_velocity(current_pos: Tuple[float, float], 
                     previous_pos: Tuple[float, float],
                     dt: float = 1.0) -> Tuple[float, float]:
    """
    Compute velocity vector from position change.
    
    Args:
        current_pos: Current (x, y) position
        previous_pos: Previous (x, y) position
        dt: Time step (default: 1 frame)
    
    Returns:
        (vx, vy) velocity vector
    """
    vx = (current_pos[0] - previous_pos[0]) / dt
    vy = (current_pos[1] - previous_pos[1]) / dt
    return (vx, vy)


def compute_velocity_magnitude(current_pos: Tuple[float, float],
                               previous_pos: Tuple[float, float],
                               dt: float = 1.0) -> float:
    """
    Compute velocity magnitude (speed) from position change.
    
    Args:
        current_pos: Current (x, y) position
        previous_pos: Previous (x, y) position
        dt: Time step (default: 1 frame)
    
    Returns:
        Speed magnitude
    """
    vx, vy = compute_velocity(current_pos, previous_pos, dt)
    return np.sqrt(vx**2 + vy**2)


def compute_heading(current_pos: Tuple[float, float],
                    previous_pos: Tuple[float, float]) -> float:
    """
    Compute heading angle (direction) from position change.
    
    Args:
        current_pos: Current (x, y) position
        previous_pos: Previous (x, y) position
    
    Returns:
        Heading angle in radians (0 to 2*pi)
    """
    dx = current_pos[0] - previous_pos[0]
    dy = current_pos[1] - previous_pos[1]
    
    # Avoid division by zero
    if dx == 0 and dy == 0:
        return 0.0
    
    heading = np.arctan2(dy, dx)
    # Normalize to [0, 2*pi]
    if heading < 0:
        heading += 2 * np.pi
    return heading


def compute_heading_alignment(heading1: float, heading2: float) -> float:
    """
    Compute alignment between two headings (0 = opposite, 1 = same direction).
    
    Args:
        heading1: First heading angle in radians
        heading2: Second heading angle in radians
    
    Returns:
        Alignment score in [0, 1] where 1 means same direction, 0 means opposite
    """
    # Compute angle difference
    diff = abs(heading1 - heading2)
    # Normalize to [0, pi]
    if diff > np.pi:
        diff = 2 * np.pi - diff
    
    # Convert to alignment score: 1 when same direction (diff=0), 0 when opposite (diff=pi)
    alignment = 1.0 - (diff / np.pi)
    return alignment


def normalize_distance(distance: float, max_distance: float = 100.0) -> float:
    """
    Normalize distance to [0, 1] range.
    
    Args:
        distance: Raw distance value
        max_distance: Maximum expected distance (default: 100 pixels)
    
    Returns:
        Normalized distance in [0, 1] (1 = very close, 0 = far away)
    """
    # Clamp distance to max_distance
    distance = min(distance, max_distance)
    # Normalize: closer = higher value (inverse relationship)
    normalized = 1.0 - (distance / max_distance)
    return max(0.0, normalized)


def normalize_velocity_difference(vel_diff: float, max_vel_diff: float = 10.0) -> float:
    """
    Normalize velocity difference to [0, 1] range.
    
    Args:
        vel_diff: Absolute velocity difference
        max_vel_diff: Maximum expected velocity difference (default: 10 pixels/frame)
    
    Returns:
        Normalized velocity difference in [0, 1] (1 = high difference, 0 = no difference)
    """
    # Clamp to max_vel_diff
    vel_diff = min(vel_diff, max_vel_diff)
    # Normalize: higher difference = higher value
    normalized = vel_diff / max_vel_diff
    return normalized


def compute_relational_features(
    target_pos: Tuple[float, float],
    target_prev_pos: Optional[Tuple[float, float]],
    obstacle_pos: Tuple[float, float],
    obstacle_prev_pos: Optional[Tuple[float, float]],
    max_distance: float = 100.0,
    max_vel_diff: float = 10.0
) -> Tuple[float, float, float, float]:
    """
    Compute 4 relational features between target and obstacle.
    
    Args:
        target_pos: Current position of target (x, y)
        target_prev_pos: Previous position of target (x, y) or None
        obstacle_pos: Current position of obstacle (x, y)
        obstacle_prev_pos: Previous position of obstacle (x, y) or None
        max_distance: Maximum distance for normalization
        max_vel_diff: Maximum velocity difference for normalization
    
    Returns:
        Tuple of (f1, f2, f3, f4):
        - f1: Normalized distance feature (higher = closer)
        - f2: Normalized velocity difference feature (higher = more different)
        - f3: Heading alignment feature (higher = more aligned)
        - f4: Object class interaction feature (placeholder, currently based on distance)
    """
    # Feature 1: Distance
    distance = compute_distance(target_pos, obstacle_pos)
    f1 = normalize_distance(distance, max_distance)
    
    # Feature 2: Velocity difference
    if target_prev_pos is not None and obstacle_prev_pos is not None:
        target_vel = compute_velocity_magnitude(target_pos, target_prev_pos)
        obstacle_vel = compute_velocity_magnitude(obstacle_pos, obstacle_prev_pos)
        vel_diff = abs(target_vel - obstacle_vel)
        f2 = normalize_velocity_difference(vel_diff, max_vel_diff)
    else:
        # If no previous position, set to 0 (no velocity difference)
        f2 = 0.0
    
    # Feature 3: Heading alignment
    if target_prev_pos is not None and obstacle_prev_pos is not None:
        target_heading = compute_heading(target_pos, target_prev_pos)
        obstacle_heading = compute_heading(obstacle_pos, obstacle_prev_pos)
        f3 = compute_heading_alignment(target_heading, obstacle_heading)
    else:
        # If no previous position, set to 0.5 (neutral alignment)
        f3 = 0.5
    
    # Feature 4: Object class interaction (placeholder)
    # For now, use inverse distance as a simple proxy
    # In the future, this could incorporate actual object class information
    f4 = normalize_distance(distance, max_distance) * 0.5  # Scale down as placeholder
    
    return (f1, f2, f3, f4)


def compute_threat_score(
    f1: float,
    f2: float,
    f3: float,
    f4: float,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0,
    w4: float = 1.0
) -> float:
    """
    Compute threat score using weighted sum and sigmoid.
    
    Args:
        f1: Distance feature
        f2: Velocity difference feature
        f3: Heading alignment feature
        f4: Object class interaction feature
        w1, w2, w3, w4: Weights for each feature (default: 1.0 each)
    
    Returns:
        Threat score in [0, 1] range
    """
    # Weighted sum
    weighted_sum = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4
    
    # Apply sigmoid to keep output in [0, 1]
    # Using standard sigmoid: 1 / (1 + exp(-x))
    threat_score = 1.0 / (1.0 + np.exp(-weighted_sum))
    
    return threat_score


def compute_threat_scores_for_frame(
    frame_id: int,
    target_id: int,
    frame_data: Dict[int, List[Tuple[int, float, float]]],
    target_positions: Dict[int, Tuple[float, float]],
    object_positions: Dict[int, Dict[int, Tuple[float, float]]],
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    max_distance: float = 100.0,
    max_vel_diff: float = 10.0
) -> List[Tuple[int, float, Tuple[float, float], Tuple[float, float, float, float]]]:
    """
    Compute threat scores for all obstacles relative to target in a frame.
    
    Args:
        frame_id: Current frame ID
        target_id: Target pedestrian ID
        frame_data: Dictionary mapping frame_id to list of (ped_id, x, y) tuples
        target_positions: Dictionary mapping frame_id to target (x, y) position
        object_positions: Dictionary mapping object_id to dict of frame_id -> (x, y)
        weights: Tuple of (w1, w2, w3, w4) weights
        max_distance: Maximum distance for normalization
        max_vel_diff: Maximum velocity difference for normalization
    
    Returns:
        List of tuples (obstacle_id, threat_score, obstacle_position, features)
        where features is (f1, f2, f3, f4)
    """
    if frame_id not in frame_data:
        return []
    
    # Get target position
    if frame_id not in target_positions:
        return []
    
    target_pos = target_positions[frame_id]
    target_prev_pos = target_positions.get(frame_id - 1)
    
    # Get all objects in this frame
    objects_in_frame = frame_data[frame_id]
    
    threat_scores = []
    w1, w2, w3, w4 = weights
    
    for obstacle_id, obstacle_x, obstacle_y in objects_in_frame:
        # Skip the target itself
        if obstacle_id == target_id:
            continue
        
        obstacle_pos = (obstacle_x, obstacle_y)
        
        # Get previous position of obstacle
        obstacle_prev_pos = None
        if obstacle_id in object_positions:
            obstacle_prev_pos = object_positions[obstacle_id].get(frame_id - 1)
        
        # Compute relational features
        f1, f2, f3, f4 = compute_relational_features(
            target_pos, target_prev_pos,
            obstacle_pos, obstacle_prev_pos,
            max_distance=max_distance,
            max_vel_diff=max_vel_diff
        )
        
        # Compute threat score
        threat_score = compute_threat_score(f1, f2, f3, f4, w1, w2, w3, w4)
        
        threat_scores.append((
            obstacle_id,
            threat_score,
            obstacle_pos,
            (f1, f2, f3, f4)
        ))
    
    return threat_scores


def build_object_position_history(
    annotation_path: str
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Build a dictionary mapping object_id to frame_id -> position.
    
    Args:
        annotation_path: Path to annotation file
    
    Returns:
        Dictionary mapping object_id to dict of frame_id -> (x, y) position
    """
    frame_data = get_object_positions_per_frame(annotation_path)
    
    object_positions = defaultdict(dict)
    
    for frame_id, objects in frame_data.items():
        for ped_id, x, y in objects:
            object_positions[ped_id][frame_id] = (x, y)
    
    return dict(object_positions)

