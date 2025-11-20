#!/usr/bin/env python3
"""
Debug script to check FOV calculation for obstacles.
"""

import numpy as np
from threat_score_viz.threat_score_computer import (
    compute_heading,
    compute_angle_to_obstacle,
    compute_angle_difference,
    is_obstacle_in_field_of_view
)

# Test case: Target moving east, obstacle behind (west)
target_pos = (10.0, 10.0)
target_prev_pos = (9.0, 10.0)  # Moving east
obstacle_pos = (8.0, 10.0)  # Behind target (west)

print("Test Case: Target moving east, obstacle behind")
print(f"Target pos: {target_pos}")
print(f"Target prev pos: {target_prev_pos}")
print(f"Obstacle pos: {obstacle_pos}")
print()

# Compute heading
target_heading = compute_heading(target_pos, target_prev_pos)
print(f"Target heading: {target_heading:.3f} radians ({np.degrees(target_heading):.1f} degrees)")
print(f"  (0 = east, π/2 = north, π = west, 3π/2 = south)")

# Compute angle to obstacle
angle_to_obstacle = compute_angle_to_obstacle(target_pos, obstacle_pos)
print(f"Angle to obstacle: {angle_to_obstacle:.3f} radians ({np.degrees(angle_to_obstacle):.1f} degrees)")

# Compute angle difference
angle_diff = compute_angle_difference(target_heading, angle_to_obstacle)
print(f"Angle difference: {angle_diff:.3f} radians ({np.degrees(angle_diff):.1f} degrees)")

# Check FOV (100 degrees = 1.745 radians)
fov_angle = np.pi * 100 / 180.0
half_fov = fov_angle / 2.0
print(f"\nFOV: {np.degrees(fov_angle):.1f} degrees ({fov_angle:.3f} radians)")
print(f"Half FOV: {np.degrees(half_fov):.1f} degrees ({half_fov:.3f} radians)")
print(f"Angle diff <= half_fov? {angle_diff <= half_fov}")

is_visible = is_obstacle_in_field_of_view(target_pos, target_prev_pos, obstacle_pos, fov_angle)

# Compute dot product for debugging
dx = target_pos[0] - target_prev_pos[0]
dy = target_pos[1] - target_prev_pos[1]
vel_mag = np.sqrt(dx**2 + dy**2)
vel_x = dx / vel_mag
vel_y = dy / vel_mag
rel_x = obstacle_pos[0] - target_pos[0]
rel_y = obstacle_pos[1] - target_pos[1]
rel_mag = np.sqrt(rel_x**2 + rel_y**2)
rel_x_norm = rel_x / rel_mag
rel_y_norm = rel_y / rel_mag
dot = vel_x * rel_x_norm + vel_y * rel_y_norm
cos_threshold = np.cos(fov_angle / 2.0)
print(f"\nDot product: {dot:.3f} (threshold: {cos_threshold:.3f})")
print(f"Is visible? {is_visible}")
print(f"Expected: False (obstacle is behind)")

print("\n" + "="*60)

# Test case: Target moving east, obstacle in front (east)
obstacle_pos2 = (12.0, 10.0)  # In front of target (east)
print("\nTest Case: Target moving east, obstacle in front")
print(f"Obstacle pos: {obstacle_pos2}")

angle_to_obstacle2 = compute_angle_to_obstacle(target_pos, obstacle_pos2)
angle_diff2 = compute_angle_difference(target_heading, angle_to_obstacle2)
print(f"Angle to obstacle: {angle_to_obstacle2:.3f} radians ({np.degrees(angle_to_obstacle2):.1f} degrees)")
print(f"Angle difference: {angle_diff2:.3f} radians ({np.degrees(angle_diff2):.1f} degrees)")
print(f"Angle diff <= half_fov? {angle_diff2 <= half_fov}")

is_visible2 = is_obstacle_in_field_of_view(target_pos, target_prev_pos, obstacle_pos2, fov_angle)
print(f"Is visible? {is_visible2}")
print(f"Expected: True (obstacle is in front)")

print("\n" + "="*60)

# Test case: Target moving east, obstacle to the side (north)
obstacle_pos3 = (10.0, 12.0)  # To the side (north)
print("\nTest Case: Target moving east, obstacle to the side (north)")
print(f"Obstacle pos: {obstacle_pos3}")

angle_to_obstacle3 = compute_angle_to_obstacle(target_pos, obstacle_pos3)
angle_diff3 = compute_angle_difference(target_heading, angle_to_obstacle3)
print(f"Angle to obstacle: {angle_to_obstacle3:.3f} radians ({np.degrees(angle_to_obstacle3):.1f} degrees)")
print(f"Angle difference: {angle_diff3:.3f} radians ({np.degrees(angle_diff3):.1f} degrees)")
print(f"Angle diff <= half_fov? {angle_diff3 <= half_fov}")

is_visible3 = is_obstacle_in_field_of_view(target_pos, target_prev_pos, obstacle_pos3, fov_angle)
print(f"Is visible? {is_visible3}")
print(f"Expected: True (obstacle is to the side, within 100-degree FOV)")

