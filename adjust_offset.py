#!/usr/bin/env python3
"""
Adjust offset to fix alignment. Since movements match, scale is correct,
we just need to shift everything by the right amount.
"""

import cv2
import numpy as np
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.visualizer import calculate_coordinate_transform, transform_coordinates

video_path = 'sdd_bookstore/bookstore_vid/video0/video.mp4'
annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
frame_id = 5000

# Load video frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
print(f'Video: {w}x{h}')

# Get annotations
frame_data = get_object_positions_per_frame(annotation_path)
if frame_id not in frame_data:
    print(f'Frame {frame_id} not found')
    exit(1)

# Calculate base transformation (scale is correct, offset needs adjustment)
coord_transform = calculate_coordinate_transform(annotation_path, w, h, preserve_position=True)
scale_x, scale_y, base_offset_x, base_offset_y = coord_transform

print(f'\nCurrent transformation:')
print(f'  Scale: ({scale_x:.2f}, {scale_y:.2f}) - KEEP THIS (movements match)')
print(f'  Base offset: ({base_offset_x:.2f}, {base_offset_y:.2f}) - ADJUST THIS')
print()

# Try different offset adjustments
# Since objects are shifted, we need to find the offset correction
offset_adjustments = [
    (0, 0, 'No adjustment'),
    (-200, -200, 'Shift left and up by 200px'),
    (-300, -300, 'Shift left and up by 300px'),
    (-400, -400, 'Shift left and up by 400px'),
    (-500, -500, 'Shift left and up by 500px'),
    (200, 200, 'Shift right and down by 200px'),
    (300, 300, 'Shift right and down by 300px'),
    (-100, 100, 'Shift left 100px, down 100px'),
    (100, -100, 'Shift right 100px, up 100px'),
]

print('Testing different offset adjustments...')
print('(Scale stays the same, only offset changes)')
print()

for adj_x, adj_y, desc in offset_adjustments:
    test_frame = frame.copy()
    offset_x = base_offset_x + adj_x
    offset_y = base_offset_y + adj_y
    
    # Draw objects with this offset
    for obj_id, x, y in frame_data[frame_id][:15]:
        px = x * scale_x + offset_x
        py = y * scale_y + offset_y
        px, py = int(px), int(py)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(test_frame, (px, py), 12, (0, 255, 0), 2)
            cv2.putText(test_frame, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    filename = f'output/offset_test_{adj_x}_{adj_y}.png'
    cv2.imwrite(filename, test_frame)
    print(f'  {desc}: offset=({offset_x:.0f}, {offset_y:.0f}) -> {filename}')

print('\n' + '=' * 60)
print('Check the offset_test_*.png images to find which offset aligns best.')
print('Once you find the best one, we can update the transformation.')
print('=' * 60)

# Also create a version where you can manually specify offset
print('\nTo manually specify offset:')
print('1. Look at the test images above')
print('2. Find which offset aligns best with objects')
print('3. Or provide the offset adjustment needed (e.g., \"shift left 300px, up 200px\")')
print('4. We can then update the transformation function with the corrected offset')

