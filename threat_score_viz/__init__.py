"""
Threat Score Visualization Package

This package provides tools for computing and visualizing threat scores
for pedestrian trajectories in the SDD dataset.
"""

from .data_parser import (
    load_annotations,
    organize_by_frame,
    get_object_positions_per_frame,
    get_available_object_ids,
    get_frame_range
)

from .video_utils import (
    get_video_properties,
    verify_annotation_video_alignment,
    load_video_frame
)

from .target_selector import (
    get_object_presence_stats,
    get_object_average_position,
    find_best_target_candidates,
    find_central_target_candidates,
    validate_target,
    get_target_positions,
    select_target
)

from .threat_score_computer import (
    compute_distance,
    compute_velocity,
    compute_heading,
    compute_angle_to_obstacle,
    compute_angle_difference,
    compute_smoothed_velocity_vector,
    is_obstacle_in_field_of_view,
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

from .visualizer import (
    draw_threat_score_on_frame,
    draw_target_marker,
    process_video_frame,
    create_frame_metadata,
    process_video_with_threat_scores
)

__all__ = [
    'load_annotations',
    'organize_by_frame',
    'get_object_positions_per_frame',
    'get_available_object_ids',
    'get_frame_range',
    'get_video_properties',
    'verify_annotation_video_alignment',
    'load_video_frame',
    'get_object_presence_stats',
    'get_object_average_position',
    'find_best_target_candidates',
    'find_central_target_candidates',
    'validate_target',
    'get_target_positions',
    'select_target',
    'compute_distance',
    'compute_velocity',
    'compute_heading',
    'compute_angle_to_obstacle',
    'compute_angle_difference',
    'compute_smoothed_velocity_vector',
    'is_obstacle_in_field_of_view',
    'compute_relative_vectors',
    'compute_approach_velocity',
    'compute_ttc',
    'compute_threat_features',
    'normalize_threat_features_minmax',
    'compute_threat_score',
    'compute_threat_scores_for_frame',
    'build_object_position_history',
    'get_obstacle_size',
    'draw_threat_score_on_frame',
    'draw_target_marker',
    'process_video_frame',
    'create_frame_metadata',
    'process_video_with_threat_scores',
]

