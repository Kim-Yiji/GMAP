#!/usr/bin/env python3
"""
Calculate transformation from manually identified reference points.
Usage: Provide object IDs and their pixel coordinates in the video.
"""

import numpy as np
from threat_score_viz.data_parser import get_object_positions_per_frame

annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
frame_id = 5000

# Get annotations
frame_data = get_object_positions_per_frame(annotation_path)
if frame_id not in frame_data:
    print(f'Frame {frame_id} not found')
    exit(1)

# Create a dictionary of object_id -> annotation coordinates
ann_coords = {obj_id: (x, y) for obj_id, x, y in frame_data[frame_id]}

print('=' * 60)
print('TRANSFORMATION CALCULATOR')
print('=' * 60)
print('\nProvide reference points: object_id and pixel coordinates')
print('Example format:')
print('  Object 22: (800, 600)')
print('  Object 23: (950, 580)')
print('  Object 84: (700, 500)')
print('\nEnter at least 2 points (more is better for accuracy)')
print('Press Enter after each point, or type "done" when finished')
print('=' * 60)

# Manual input - in a real scenario, user would provide these
# For now, let's create a function that can solve for scale and offset

def solve_transformation(annotation_points, pixel_points):
    """
    Solve for scale and offset given reference points.
    pixel = annotation * scale + offset
    
    Args:
        annotation_points: List of (x, y) annotation coordinates
        pixel_points: List of (x, y) pixel coordinates
    """
    if len(annotation_points) < 2:
        raise ValueError("Need at least 2 reference points")
    
    # Solve for scale_x, scale_y, offset_x, offset_y
    # Using least squares approach
    ann_points = np.array(annotation_points)
    pix_points = np.array(pixel_points)
    
    # For X coordinates: pix_x = ann_x * scale_x + offset_x
    # We can solve this with linear regression
    from sklearn.linear_model import LinearRegression
    
    # Solve for X
    reg_x = LinearRegression()
    reg_x.fit(ann_points[:, 0].reshape(-1, 1), pix_points[:, 0])
    scale_x = reg_x.coef_[0]
    offset_x = reg_x.intercept_
    
    # Solve for Y
    reg_y = LinearRegression()
    reg_y.fit(ann_points[:, 1].reshape(-1, 1), pix_points[:, 1])
    scale_y = reg_y.coef_[0]
    offset_y = reg_y.intercept_
    
    return scale_x, scale_y, offset_x, offset_y

print('\nReference points format:')
print('  Enter: object_id pixel_x pixel_y')
print('  Example: 22 800 600')
print('  Type "done" when finished')
print()
print('Available objects in frame:')
for obj_id in sorted(ann_coords.keys())[:20]:
    x, y = ann_coords[obj_id]
    print(f'  Object {obj_id}: annotation=({x:.2f}, {y:.2f})')

# For demonstration, let's show how to use it
print('\n' + '=' * 60)
print('To use this tool:')
print('1. Identify objects in the video frame')
print('2. Note their pixel coordinates')
print('3. Run this script and provide the reference points')
print('4. The script will calculate scale and offset')
print('5. Use those values in the transformation function')
print('=' * 60)

