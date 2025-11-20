#!/usr/bin/env python3
"""
Debug FOV calculation with real video data.
"""

import numpy as np
from threat_score_viz.threat_score_computer import (
    compute_heading,
    compute_angle_to_obstacle,
    compute_angle_difference,
    is_obstacle_in_field_of_view,
    compute_threat_scores_for_frame,
    build_object_position_history
)
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.target_selector import get_target_positions

annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
target_id = 212
frame_id = 5000

# Load data
frame_data = get_object_positions_per_frame(annotation_path)
target_positions = get_target_positions(annotation_path, target_id)
object_positions = build_object_position_history(annotation_path)

if frame_id not in target_positions:
    print(f"Target not in frame {frame_id}")
    exit(1)

target_pos = target_positions[frame_id]
target_prev_pos = target_positions.get(frame_id - 1)

print(f"Frame {frame_id}")
print(f"Target ID: {target_id}")
print(f"Target pos: {target_pos}")
print(f"Target prev pos: {target_prev_pos}")
print()

if target_prev_pos:
    dx = target_pos[0] - target_prev_pos[0]
    dy = target_pos[1] - target_prev_pos[1]
    movement = np.sqrt(dx**2 + dy**2)
    heading = compute_heading(target_pos, target_prev_pos)
    print(f"Movement: ({dx:.3f}, {dy:.3f}), distance: {movement:.3f}")
    print(f"Heading: {heading:.3f} radians ({np.degrees(heading):.1f} degrees)")
    print(f"  (0 = east, π/2 = north, π = west, 3π/2 = south)")
else:
    print("No previous position")
    exit(1)

print("\n" + "="*80)
print("Obstacles in frame:")
print("="*80)

fov_angle = np.pi * 100 / 180.0
half_fov = fov_angle / 2.0

if frame_id in frame_data:
    objects = frame_data[frame_id]
    for obstacle_id, obstacle_x, obstacle_y in objects:
        if obstacle_id == target_id:
            continue
        
        obstacle_pos = (obstacle_x, obstacle_y)
        
        # Compute angles
        angle_to_obstacle = compute_angle_to_obstacle(target_pos, obstacle_pos)
        angle_diff = compute_angle_difference(heading, angle_to_obstacle)
        
        # Check visibility
        is_visible = is_obstacle_in_field_of_view(
            target_pos, target_prev_pos, obstacle_pos, fov_angle
        )
        
        # Compute relative position
        rel_x = obstacle_pos[0] - target_pos[0]
        rel_y = obstacle_pos[1] - target_pos[1]
        distance = np.sqrt(rel_x**2 + rel_y**2)
        
        # Determine position relative to heading
        if angle_diff <= half_fov:
            position_desc = "IN FOV"
        elif angle_diff > np.pi - half_fov:
            position_desc = "BEHIND"
        else:
            position_desc = "OUTSIDE FOV"
        
        print(f"\nObstacle {obstacle_id}:")
        print(f"  Position: ({obstacle_x:.2f}, {obstacle_y:.2f})")
        print(f"  Relative: ({rel_x:.2f}, {rel_y:.2f}), distance: {distance:.2f}")
        print(f"  Angle to obstacle: {np.degrees(angle_to_obstacle):.1f}°")
        print(f"  Angle difference: {np.degrees(angle_diff):.1f}° (half FOV: {np.degrees(half_fov):.1f}°)")
        print(f"  Position: {position_desc}")
        print(f"  Is visible: {is_visible} (should be {angle_diff <= half_fov})")
        
        if angle_diff > half_fov and is_visible:
            print(f"  ⚠ BUG: Should be grey but is visible!")
        elif angle_diff <= half_fov and not is_visible:
            print(f"  ⚠ BUG: Should be visible but is grey!")

