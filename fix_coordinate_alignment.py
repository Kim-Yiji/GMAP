#!/usr/bin/env python3
"""
Diagnostic script to check coordinate alignment and suggest the best approach.
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
    print(f'Frame {frame_id} not found in annotations')
    exit(1)

# Test 1: Raw coordinates (no transformation)
print('\n=== Test 1: Using annotations as pixel coordinates directly ===')
test_frame1 = frame.copy()
for obj_id, x, y in frame_data[frame_id][:10]:
    px, py = int(x), int(y)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame1, (px, py), 15, (0, 255, 0), 3)  # Green
        cv2.putText(test_frame1, f'{obj_id}', (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f'  Object {obj_id}: annotation=({x:.1f}, {y:.1f}) -> pixel=({px}, {py})')
cv2.imwrite('output/test_raw_coords.png', test_frame1)
print('  Saved: output/test_raw_coords.png (green circles = raw coordinates)')

# Test 2: Transformed with centering
print('\n=== Test 2: Transformed with centering (old method) ===')
test_frame2 = frame.copy()
coord_transform_center = calculate_coordinate_transform(annotation_path, w, h, preserve_position=False)
scale_x, scale_y, offset_x, offset_y = coord_transform_center
print(f'  Transformation: scale=({scale_x:.2f}, {scale_y:.2f}), offset=({offset_x:.2f}, {offset_y:.2f})')
for obj_id, x, y in frame_data[frame_id][:10]:
    px, py = transform_coordinates(x, y, scale_x, scale_y, offset_x, offset_y)
    px, py = int(px), int(py)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame2, (px, py), 15, (255, 0, 0), 3)  # Blue
        cv2.putText(test_frame2, f'{obj_id}', (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f'  Object {obj_id}: annotation=({x:.1f}, {y:.1f}) -> pixel=({px}, {py})')
cv2.imwrite('output/test_centered_coords.png', test_frame2)
print('  Saved: output/test_centered_coords.png (blue circles = centered transformation)')

# Test 3: Transformed preserving position (new method)
print('\n=== Test 3: Transformed preserving position (new method) ===')
test_frame3 = frame.copy()
coord_transform_preserve = calculate_coordinate_transform(annotation_path, w, h, preserve_position=True)
scale_x, scale_y, offset_x, offset_y = coord_transform_preserve
print(f'  Transformation: scale=({scale_x:.2f}, {scale_y:.2f}), offset=({offset_x:.2f}, {offset_y:.2f})')
for obj_id, x, y in frame_data[frame_id][:10]:
    px, py = transform_coordinates(x, y, scale_x, scale_y, offset_x, offset_y)
    px, py = int(px), int(py)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame3, (px, py), 15, (0, 0, 255), 3)  # Red
        cv2.putText(test_frame3, f'{obj_id}', (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f'  Object {obj_id}: annotation=({x:.1f}, {y:.1f}) -> pixel=({px}, {py})')
cv2.imwrite('output/test_preserved_coords.png', test_frame3)
print('  Saved: output/test_preserved_coords.png (red circles = preserved position transformation)')

print('\n=== Recommendation ===')
print('Compare the three output images to see which method aligns best with actual people:')
print('1. test_raw_coords.png - If green circles align with people, use --no-coord-transform')
print('2. test_centered_coords.png - If blue circles align, use centered transformation')
print('3. test_preserved_coords.png - If red circles align, use preserved position (current default)')
print('\nThe gap issue is likely because:')
print('- Annotations might already be in pixel coordinates (use method 1)')
print('- OR the transformation needs adjustment (methods 2 or 3)')
print('- OR SDD needs a homography matrix (not available)')

