"""
Threat score computation module using the DMRGCN threat score algorithm.

This module computes threat scores based on 4 relational features:
- d_ij: Distance between target and obstacle
- v+_ij: Approach velocity (component of relative velocity pointing toward target)
- size_j: Obstacle size (0.0=human, 0.2=similar, 0.7=larger, 1.0=much larger)
- TTC_ij: Time-to-Collision

Pipeline:
1. Compute raw features (d, v+, size, TTC)
2. Min-max normalization with direction reversal for d and TTC
3. Linear weighted combination: u = w_d*d' + w_v*v+' + w_s*size' + w_T*TTC'
4. Sigmoid transformation: Threat = sigmoid((u - beta) / tau)

Default weights: [0.5, 0.25, 0.15, 0.1] (distance-weighted)
Default parameters: tau=0.15, beta=0.5
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .data_parser import get_object_positions_per_frame
from .target_selector import get_target_positions


def get_obstacle_size(obstacle_type=None, default_size=0.0):
    """
    Get obstacle size value based on obstacle type.
    
    Args:
        obstacle_type: Type of obstacle (string or int identifier)
        default_size: Default size if type is unknown (default: 0.0 for human size)
    
    Returns:
        size_j: Scalar size value
            - 0.0: Human size (pedestrian, skateboarder, person pushing stroller)
            - 0.2: Similar to human size (bicycle, shopping cart)
            - 0.7: Larger than human (car)
            - 1.0: Much larger than human (truck, bus, train)
    """
    if obstacle_type is None:
        return default_size
    
    # Map obstacle types to size values
    size_mapping = {
        # Human size (0.0)
        'pedestrian': 0.0,
        'person': 0.0,
        'skateboarder': 0.0,
        'skater': 0.0,  # SDD dataset
        'stroller': 0.0,  # Person pushing stroller (human-sized)
        # Similar to human size (0.2)
        'bicycle': 0.2,
        'bike': 0.2,
        'biker': 0.2,  # SDD dataset - bicycle rider (similar to human)
        'shopping_cart': 0.2,
        'cart': 0.2,  # SDD dataset
        # Larger than human (0.7)
        'car': 0.7,  # SDD dataset
        'vehicle': 0.7,
        # Much larger than human (1.0)
        'truck': 1.0,
        'bus': 1.0,  # SDD dataset
        'train': 1.0,
    }
    
    # Handle string input
    if isinstance(obstacle_type, str):
        obstacle_type_lower = obstacle_type.lower()
        return size_mapping.get(obstacle_type_lower, default_size)
    
    # Handle integer/other types - assume default (human size)
    return default_size


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


def compute_heading(current_pos: Tuple[float, float],
                    previous_pos: Tuple[float, float]) -> float:
    """
    Compute heading angle (direction of movement) from position change.
    
    Args:
        current_pos: Current (x, y) position
        previous_pos: Previous (x, y) position
    
    Returns:
        Heading angle in radians (0 to 2*pi), where 0 is east (positive x)
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


def compute_angle_to_obstacle(
    target_pos: Tuple[float, float],
    obstacle_pos: Tuple[float, float]
) -> float:
    """
    Compute angle from target to obstacle.
    
    Args:
        target_pos: Position of target (x, y)
        obstacle_pos: Position of obstacle (x, y)
    
    Returns:
        Angle in radians (0 to 2*pi), where 0 is east (positive x)
    """
    dx = obstacle_pos[0] - target_pos[0]
    dy = obstacle_pos[1] - target_pos[1]
    
    # Avoid division by zero
    if dx == 0 and dy == 0:
        return 0.0
    
    angle = np.arctan2(dy, dx)
    # Normalize to [0, 2*pi]
    if angle < 0:
        angle += 2 * np.pi
    return angle


def compute_angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute the smallest angle difference between two angles.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
    
    Returns:
        Angle difference in radians [0, pi]
    """
    diff = abs(angle1 - angle2)
    # Normalize to [0, pi]
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def compute_smoothed_velocity_vector(
    target_positions: List[Tuple[float, float]],
    min_movement_threshold: float = 0.1
) -> Optional[Tuple[float, float]]:
    """
    Compute smoothed velocity vector from multiple previous positions.
    
    Uses weighted average of velocity vectors from recent frames to reduce noise.
    More recent frames have higher weight.
    
    Args:
        target_positions: List of (x, y) positions from most recent to oldest
            [current, frame-1, frame-2, ...]
        min_movement_threshold: Minimum total movement across all frames to compute heading
    
    Returns:
        Normalized velocity vector (vx, vy) or None if insufficient movement
    """
    if len(target_positions) < 2:
        return None
    
    # Compute velocity vectors for each consecutive pair
    velocities = []
    total_movement = 0.0
    
    for i in range(len(target_positions) - 1):
        pos_curr = target_positions[i]
        pos_prev = target_positions[i + 1]
        
        dx = pos_curr[0] - pos_prev[0]
        dy = pos_curr[1] - pos_prev[1]
        movement = np.sqrt(dx**2 + dy**2)
        
        if movement > 1e-6:  # Avoid division by zero
            # Normalize to get direction
            vx = dx / movement
            vy = dy / movement
            velocities.append((vx, vy, movement))
            total_movement += movement
    
    if total_movement < min_movement_threshold or len(velocities) == 0:
        return None
    
    # Weighted average: more recent frames have higher weight
    # Weight = movement_distance * recency_factor
    # recency_factor decreases for older frames
    weighted_vx = 0.0
    weighted_vy = 0.0
    total_weight = 0.0
    
    for i, (vx, vy, movement) in enumerate(velocities):
        # Recency weight: most recent frame gets weight 1.0, older frames get less
        recency_weight = 1.0 / (i + 1.0)
        weight = movement * recency_weight
        
        weighted_vx += vx * weight
        weighted_vy += vy * weight
        total_weight += weight
    
    if total_weight < 1e-6:
        return None
    
    # Normalize the weighted average
    vx_avg = weighted_vx / total_weight
    vy_avg = weighted_vy / total_weight
    
    # Normalize to unit vector
    magnitude = np.sqrt(vx_avg**2 + vy_avg**2)
    if magnitude < 1e-6:
        return None
    
    return (vx_avg / magnitude, vy_avg / magnitude)


def is_obstacle_in_field_of_view(
    target_pos: Tuple[float, float],
    target_prev_pos: Optional[Tuple[float, float]],
    obstacle_pos: Tuple[float, float],
    fov_angle: float = np.pi / 2,  # 90 degrees by default
    min_movement_threshold: float = 0.05,  # Minimum movement to use heading-based FOV
    target_position_history: Optional[List[Tuple[float, float]]] = None  # For smoothed heading
) -> bool:
    """
    Check if obstacle is within target's field of view based on target's movement direction.
    
    Uses smoothed velocity vector (from multiple frames) if available, otherwise falls back
    to single-frame velocity. Uses dot product to determine if obstacle is in front (visible)
    or behind (not visible).
    
    Args:
        target_pos: Current position of target (x, y)
        target_prev_pos: Previous position of target (x, y) or None
        obstacle_pos: Position of obstacle (x, y)
        fov_angle: Field of view angle in radians (default: pi/2 = 90 degrees)
        min_movement_threshold: Minimum movement distance to use heading-based FOV.
            If target moved less than this, assume they can see in all directions.
        target_position_history: Optional list of recent target positions [current, frame-1, ...]
            for smoothed heading calculation. If provided, uses smoothed heading.
    
    Returns:
        True if obstacle is within field of view, False otherwise
    """
    # Try to use smoothed heading if history is available
    if target_position_history is not None and len(target_position_history) >= 2:
        smoothed_vel = compute_smoothed_velocity_vector(
            target_position_history,
            min_movement_threshold=min_movement_threshold
        )
        
        if smoothed_vel is not None:
            vel_x, vel_y = smoothed_vel
        else:
            # Fall back to single-frame velocity
            if target_prev_pos is None:
                return True
            
            dx = target_pos[0] - target_prev_pos[0]
            dy = target_pos[1] - target_prev_pos[1]
            movement_distance = np.sqrt(dx**2 + dy**2)
            
            if movement_distance < min_movement_threshold:
                return True
            
            vel_x = dx / movement_distance
            vel_y = dy / movement_distance
    else:
        # Use single-frame velocity (original method)
        if target_prev_pos is None:
            return True
        
        # Compute velocity vector (movement direction)
        dx = target_pos[0] - target_prev_pos[0]
        dy = target_pos[1] - target_prev_pos[1]
        movement_distance = np.sqrt(dx**2 + dy**2)
        
        # If target moved very little, assume they can see in all directions
        if movement_distance < min_movement_threshold:
            return True
        
        # Normalize velocity vector
        vel_x = dx / movement_distance
        vel_y = dy / movement_distance
    
    # Compute relative position vector from target to obstacle
    rel_x = obstacle_pos[0] - target_pos[0]
    rel_y = obstacle_pos[1] - target_pos[1]
    rel_distance = np.sqrt(rel_x**2 + rel_y**2)
    
    # Avoid division by zero
    if rel_distance < 1e-6:
        return True
    
    # Normalize relative position vector
    rel_x_norm = rel_x / rel_distance
    rel_y_norm = rel_y / rel_distance
    
    # Compute dot product: vel · rel_pos
    # Positive = obstacle is in front (same direction as movement)
    # Negative = obstacle is behind (opposite to movement)
    dot_product = vel_x * rel_x_norm + vel_y * rel_y_norm
    
    # Convert FOV angle to cosine threshold
    # For 110-degree FOV (55 degrees on each side):
    # cos(55°) ≈ 0.574
    # Obstacle is visible if dot_product >= cos(half_fov)
    half_fov = fov_angle / 2.0
    cos_threshold = np.cos(half_fov)
    
    # If dot product is >= threshold, obstacle is within FOV
    is_visible = dot_product >= cos_threshold
    
    return is_visible


def compute_relative_vectors(
    target_pos: Tuple[float, float],
    target_prev_pos: Optional[Tuple[float, float]],
    obstacle_pos: Tuple[float, float],
    obstacle_prev_pos: Optional[Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute relative position and velocity vectors.
    
    Args:
        target_pos: Current position of target (x, y)
        target_prev_pos: Previous position of target (x, y) or None
        obstacle_pos: Current position of obstacle (x, y)
        obstacle_prev_pos: Previous position of obstacle (x, y) or None
    
    Returns:
        Tuple of (rel_pos, rel_vel):
        - rel_pos: Relative position vector r_ij = obstacle_pos - target_pos
        - rel_vel: Relative velocity vector v_ij = obstacle_vel - target_vel (or (0,0) if no previous positions)
    """
    # Relative position: r_ij = p_j - p_i
    rel_pos = (obstacle_pos[0] - target_pos[0], obstacle_pos[1] - target_pos[1])
    
    # Relative velocity: v_ij = v_j - v_i
    if target_prev_pos is not None and obstacle_prev_pos is not None:
        target_vel = compute_velocity(target_pos, target_prev_pos)
        obstacle_vel = compute_velocity(obstacle_pos, obstacle_prev_pos)
        rel_vel = (obstacle_vel[0] - target_vel[0], obstacle_vel[1] - target_vel[1])
    else:
        rel_vel = (0.0, 0.0)
    
    return rel_pos, rel_vel


def compute_approach_velocity(
    rel_pos: Tuple[float, float],
    rel_vel: Tuple[float, float],
    eps: float = 1e-6
) -> float:
    """
    Compute approach velocity v^{+}_{ij} = max(0, -r_ij · v_ij / ||r_ij||).
    
    This is the component of relative velocity pointing toward the target.
    Positive values indicate the obstacle is approaching the target.
    
    Args:
        rel_pos: Relative position vector r_ij
        rel_vel: Relative velocity vector v_ij
        eps: Small epsilon for numerical stability
    
    Returns:
        Approach velocity (non-negative)
    """
    # Compute distance ||r_ij||
    d_ij = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    d_ij_safe = d_ij + eps
    
    # Dot product: r_ij · v_ij
    dot_product = rel_pos[0] * rel_vel[0] + rel_pos[1] * rel_vel[1]
    
    # Approach velocity: v^{+}_{ij} = max(0, -r_ij · v_ij / ||r_ij||)
    v_plus = max(0.0, -dot_product / d_ij_safe)
    
    return v_plus


def compute_ttc(
    distance: float,
    approach_velocity: float,
    eps: float = 1e-6,
    ttc_max: float = 20.0
) -> float:
    """
    Compute Time-to-Collision TTC_{ij} = d_ij / max(eps, v^{+}_{ij}).
    
    Args:
        distance: Distance d_ij
        approach_velocity: Approach velocity v^{+}_{ij}
        eps: Small epsilon for numerical stability
        ttc_max: Maximum TTC value for capping (default: 20 frames)
    
    Returns:
        Time-to-collision in frames
    """
    v_plus_safe = max(eps, approach_velocity)
    ttc = distance / v_plus_safe
    
    # Cap TTC at maximum value when approach velocity is zero or very small
    if approach_velocity < eps:
        ttc = ttc_max
    else:
        ttc = min(ttc, ttc_max)
    
    return ttc


def compute_threat_features(
    target_pos: Tuple[float, float],
    target_prev_pos: Optional[Tuple[float, float]],
    obstacle_pos: Tuple[float, float],
    obstacle_prev_pos: Optional[Tuple[float, float]],
    obstacle_size: float = 0.0,
    eps: float = 1e-6,
    ttc_max: float = 20.0
) -> Tuple[float, float, float, float]:
    """
    Compute 4 threat features between target and obstacle.
    
    Args:
        target_pos: Current position of target (x, y)
        target_prev_pos: Previous position of target (x, y) or None
        obstacle_pos: Current position of obstacle (x, y)
        obstacle_prev_pos: Previous position of obstacle (x, y) or None
        obstacle_size: Size value for obstacle (default: 0.0 for human)
        eps: Small epsilon for numerical stability
        ttc_max: Maximum TTC value for capping
    
    Returns:
        Tuple of (d_ij, v+_ij, size_j, TTC_ij):
        - d_ij: Distance
        - v+_ij: Approach velocity
        - size_j: Obstacle size
        - TTC_ij: Time-to-collision
    """
    # Feature 1: Distance d_ij
    d_ij = compute_distance(target_pos, obstacle_pos)
    
    # Compute relative vectors
    rel_pos, rel_vel = compute_relative_vectors(
        target_pos, target_prev_pos,
        obstacle_pos, obstacle_prev_pos
    )
    
    # Feature 2: Approach velocity v^{+}_{ij}
    v_plus_ij = compute_approach_velocity(rel_pos, rel_vel, eps)
    
    # Feature 3: Obstacle size size_j
    size_j = obstacle_size
    
    # Feature 4: Time-to-Collision TTC_{ij}
    ttc_ij = compute_ttc(d_ij, v_plus_ij, eps, ttc_max)
    
    return (d_ij, v_plus_ij, size_j, ttc_ij)


def normalize_threat_features_minmax(
    features_list: List[Tuple[float, float, float, float]],
    reverse_direction: List[int] = [0, 3]
) -> List[Tuple[float, float, float, float]]:
    """
    Apply min-max normalization to threat features with direction reversal.
    
    Args:
        features_list: List of (d, v+, size, TTC) tuples
        reverse_direction: List of feature indices to reverse (default: [0, 3] for d and TTC)
    
    Returns:
        List of normalized feature tuples, all in [0, 1] with higher values meaning greater threat
    """
    if not features_list:
        return []
    
    # Convert to numpy array for easier computation
    features_array = np.array(features_list)  # (N, 4)
    
    normalized_features = features_array.copy()
    
    # Normalize each feature dimension separately
    for feat_idx in range(4):  # [d, v+, size, TTC]
        feat_values = features_array[:, feat_idx]
        
        if len(feat_values) == 0:
            continue
        
        feat_min = feat_values.min()
        feat_max = feat_values.max()
        
        if feat_max > feat_min:
            # Min-max normalization: x' = (x - x_min) / (x_max - x_min)
            feat_norm = (feat_values - feat_min) / (feat_max - feat_min)
            
            # Reverse direction for threat-increasing variables (d, TTC)
            # d'_ij = 1 - (d_ij - d_min) / (d_max - d_min)
            # TTC'_ij = 1 - (TTC_ij - TTC_min) / (TTC_max - TTC_min)
            if feat_idx in reverse_direction:
                feat_norm = 1.0 - feat_norm
            
            normalized_features[:, feat_idx] = feat_norm
        else:
            # All values are the same, set to 0.5 (neutral)
            normalized_features[:, feat_idx] = 0.5
    
    # Convert back to list of tuples
    return [tuple(normalized_features[i, :]) for i in range(len(features_list))]


def compute_threat_score(
    normalized_features: Tuple[float, float, float, float],
    weights: Tuple[float, float, float, float] = (0.5, 0.25, 0.15, 0.1),
    tau: float = 0.15,
    beta: float = 0.5
) -> float:
    """
    Compute threat score from normalized features using weighted sum and sigmoid.
    
    Args:
        normalized_features: Tuple of (d', v+', size', TTC') normalized features
        weights: Tuple of (w_d, w_v, w_size, w_ttc) weights (default: distance-weighted)
        tau: Temperature parameter for sigmoid (controls slope, default: 0.15)
        beta: Midpoint parameter for sigmoid (controls center, default: 0.5)
    
    Returns:
        Threat score in [0, 1] range
    """
    d_norm, v_plus_norm, size_norm, ttc_norm = normalized_features
    w_d, w_v, w_size, w_ttc = weights
    
    # Step 1: Linear weighted combination
    # u_{ij} = w_d * d'_ij + w_v * v'^+_ij + w_s * size'_j + w_T * TTC'_ij
    u_ij = w_d * d_norm + w_v * v_plus_norm + w_size * size_norm + w_ttc * ttc_norm
    
    # Step 2: Apply sigmoid transformation
    # Threat_{ij} = 1 / (1 + exp(-(u_ij - beta) / tau))
    threat_score = 1.0 / (1.0 + np.exp(-(u_ij - beta) / tau))
    
    return threat_score


def compute_threat_scores_for_frame(
    frame_id: int,
    target_id: int,
    frame_data: Dict[int, List[Tuple[int, float, float]]],
    target_positions: Dict[int, Tuple[float, float]],
    object_positions: Dict[int, Dict[int, Tuple[float, float]]],
    weights: Tuple[float, float, float, float] = (0.5, 0.25, 0.15, 0.1),
    tau: float = 0.15,
    beta: float = 0.5,
    eps: float = 1e-6,
    ttc_max: float = 20.0,
    obstacle_sizes: Optional[Dict[int, float]] = None,
    fov_angle: Optional[float] = None  # None = no FOV filtering, or angle in radians
) -> List[Tuple[int, float, Tuple[float, float], Tuple[float, float, float, float], bool]]:
    """
    Compute threat scores for all obstacles relative to target in a frame.
    
    This function computes features for all obstacles, normalizes them together,
    then computes threat scores.
    
    Args:
        frame_id: Current frame ID
        target_id: Target pedestrian ID
        frame_data: Dictionary mapping frame_id to list of (ped_id, x, y) tuples
        target_positions: Dictionary mapping frame_id to target (x, y) position
        object_positions: Dictionary mapping object_id to dict of frame_id -> (x, y)
        weights: Tuple of (w_d, w_v, w_size, w_ttc) weights
        tau: Temperature parameter for sigmoid
        beta: Midpoint parameter for sigmoid
        eps: Small epsilon for numerical stability
        ttc_max: Maximum TTC value for capping
        obstacle_sizes: Optional dictionary mapping object_id to size value
        fov_angle: Optional field of view angle in radians (None = no filtering)
    
    Returns:
        List of tuples (obstacle_id, threat_score, obstacle_position, features, is_visible)
        where features is (d_ij, v+_ij, size_j, TTC_ij) and is_visible is True if in FOV
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
    
    # Collect all obstacles and compute raw features
    obstacle_data = []
    raw_features_list = []
    
    for obstacle_id, obstacle_x, obstacle_y in objects_in_frame:
        # Skip the target itself
        if obstacle_id == target_id:
            continue
        
        obstacle_pos = (obstacle_x, obstacle_y)
        
        # Get previous position of obstacle
        obstacle_prev_pos = None
        if obstacle_id in object_positions:
            obstacle_prev_pos = object_positions[obstacle_id].get(frame_id - 1)
        
        # Get obstacle size
        obstacle_size = 0.0  # Default: human size
        if obstacle_sizes is not None and obstacle_id in obstacle_sizes:
            obstacle_size = obstacle_sizes[obstacle_id]
        
        # Compute raw features
        d_ij, v_plus_ij, size_j, ttc_ij = compute_threat_features(
            target_pos, target_prev_pos,
            obstacle_pos, obstacle_prev_pos,
            obstacle_size=obstacle_size,
            eps=eps,
            ttc_max=ttc_max
        )
        
        obstacle_data.append((obstacle_id, obstacle_pos, (d_ij, v_plus_ij, size_j, ttc_ij)))
        raw_features_list.append((d_ij, v_plus_ij, size_j, ttc_ij))
    
    if not obstacle_data:
        return []
    
    # Normalize features together (min-max normalization with direction reversal)
    normalized_features_list = normalize_threat_features_minmax(
        raw_features_list,
        reverse_direction=[0, 3]  # Reverse d and TTC
    )
    
    # Compute threat scores for each obstacle
    threat_scores = []
    for i, (obstacle_id, obstacle_pos, raw_features) in enumerate(obstacle_data):
        normalized_features = normalized_features_list[i]
        
        # Compute threat score
        threat_score = compute_threat_score(
            normalized_features,
            weights=weights,
            tau=tau,
            beta=beta
        )
        
        # Check if obstacle is in field of view
        if fov_angle is not None:
            # Build position history for smoothed heading (use up to 5 previous frames)
            position_history = [target_pos]
            for offset in range(1, 6):  # frames: -1, -2, -3, -4, -5
                prev_frame_id = frame_id - offset
                if prev_frame_id in target_positions:
                    position_history.append(target_positions[prev_frame_id])
                else:
                    break  # Stop if we hit a gap
            
            is_visible = is_obstacle_in_field_of_view(
                target_pos, target_prev_pos, obstacle_pos, fov_angle,
                target_position_history=position_history if len(position_history) >= 2 else None
            )
        else:
            is_visible = True  # No FOV filtering
        
        threat_scores.append((
            obstacle_id,
            threat_score,
            obstacle_pos,
            raw_features,  # Return raw features for metadata
            is_visible
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
