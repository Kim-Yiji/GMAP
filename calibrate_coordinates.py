#!/usr/bin/env python3
"""
Interactive calibration tool to find the correct transformation.
Since there's no homography matrix, we'll manually find where objects appear
and adjust the transformation accordingly.
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

# Strategy: Try different transformations to see which aligns best
# Since annotations are in world coordinates (0-54 range) and video is 1416x1080,
# we need to find the right scale and offset

print('\nTrying different transformation approaches...\n')

# Approach 1: Scale to fit video but keep objects in upper-left region
# (assuming objects are in upper-left of video)
print('=== Approach 1: Scale to fit, preserve upper-left position ===')
test_frame1 = frame.copy()

# Scale annotations to use ~30% of video width/height, starting from top-left
scale_factor = min(w * 0.3 / 54, h * 0.3 / 41)  # Scale to ~30% of video
offset_x = 50  # Start 50px from left
offset_y = 50  # Start 50px from top

print(f'  Scale: {scale_factor:.2f}, Offset: ({offset_x}, {offset_y})')

for obj_id, x, y in frame_data[frame_id][:15]:
    px = x * scale_factor + offset_x
    py = y * scale_factor + offset_y
    px, py = int(px), int(py)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame1, (px, py), 12, (0, 255, 255), 2)  # Yellow
        cv2.putText(test_frame1, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

cv2.imwrite('output/calibrate_approach1.png', test_frame1)
print(f'  Saved: output/calibrate_approach1.png (yellow circles)')

# Approach 2: Use larger scale, try different regions
print('\n=== Approach 2: Larger scale, upper-left region ===')
test_frame2 = frame.copy()

scale_factor2 = min(w * 0.5 / 54, h * 0.5 / 41)  # Scale to ~50% of video
offset_x2 = 100
offset_y2 = 100

print(f'  Scale: {scale_factor2:.2f}, Offset: ({offset_x2}, {offset_y2})')

for obj_id, x, y in frame_data[frame_id][:15]:
    px = x * scale_factor2 + offset_x2
    py = y * scale_factor2 + offset_y2
    px, py = int(px), int(py)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame2, (px, py), 12, (255, 255, 0), 2)  # Cyan
        cv2.putText(test_frame2, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imwrite('output/calibrate_approach2.png', test_frame2)
print(f'  Saved: output/calibrate_approach2.png (cyan circles)')

# Approach 3: Check if objects might be in a different region (center-right?)
print('\n=== Approach 3: Center-right region (where people might be) ===')
test_frame3 = frame.copy()

scale_factor3 = min(w * 0.4 / 54, h * 0.4 / 41)
offset_x3 = w * 0.3  # Start at 30% from left
offset_y3 = h * 0.2  # Start at 20% from top

print(f'  Scale: {scale_factor3:.2f}, Offset: ({offset_x3:.0f}, {offset_y3:.0f})')

for obj_id, x, y in frame_data[frame_id][:15]:
    px = x * scale_factor3 + offset_x3
    py = y * scale_factor3 + offset_y3
    px, py = int(px), int(py)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame3, (px, py), 12, (255, 0, 255), 2)  # Magenta
        cv2.putText(test_frame3, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

cv2.imwrite('output/calibrate_approach3.png', test_frame3)
print(f'  Saved: output/calibrate_approach3.png (magenta circles)')

# Approach 4: Check annotation min/max and map to a reasonable video region
print('\n=== Approach 4: Map annotation range to video region ===')
test_frame4 = frame.copy()

# Annotation range: X: 0.17-54.17, Y: 0.46-41.41
ann_min_x, ann_max_x = 0.17, 54.17
ann_min_y, ann_max_y = 0.46, 41.41
ann_range_x = ann_max_x - ann_min_x
ann_range_y = ann_max_y - ann_min_y

# Map to video region: use 40% of video, starting from (200, 150)
video_region_width = w * 0.4
video_region_height = h * 0.4
video_start_x = 200
video_start_y = 150

scale_x4 = video_region_width / ann_range_x
scale_y4 = video_region_height / ann_range_y
scale4 = min(scale_x4, scale_y4)  # Uniform scaling

offset_x4 = video_start_x - (ann_min_x * scale4)
offset_y4 = video_start_y - (ann_min_y * scale4)

print(f'  Scale: {scale4:.2f}, Offset: ({offset_x4:.0f}, {offset_y4:.0f})')
print(f'  Maps annotation range [{ann_min_x:.1f}-{ann_max_x:.1f}, {ann_min_y:.1f}-{ann_max_y:.1f}]')
print(f'  to video region starting at ({video_start_x}, {video_start_y})')

for obj_id, x, y in frame_data[frame_id][:15]:
    px = x * scale4 + offset_x4
    py = y * scale4 + offset_y4
    px, py = int(px), int(py)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(test_frame4, (px, py), 12, (0, 255, 0), 2)  # Green
        cv2.putText(test_frame4, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('output/calibrate_approach4.png', test_frame4)
print(f'  Saved: output/calibrate_approach4.png (green circles)')

print('\n=== Next Steps ===')
print('1. Check the 4 output images to see which approach aligns best with people')
print('2. Once you find the best approach, we can update the transformation function')
print('3. Or manually specify scale and offset values that work')
print('\nIf none align, we might need to:')
print('- Manually identify where a few objects appear in the video')
print('- Use those as reference points to calculate the transformation')
print('- Or check if the video/annotations are from different scenes')

