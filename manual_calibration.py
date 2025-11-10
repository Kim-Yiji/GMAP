#!/usr/bin/env python3
"""
Manual calibration: Click on objects in the video to find their pixel positions,
then calculate the transformation from annotation coordinates.
"""

import cv2
import numpy as np
from threat_score_viz.data_parser import get_object_positions_per_frame

video_path = 'sdd_bookstore/bookstore_vid/video0/video.mp4'
annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
frame_id = 5000

# Load video frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = cap.read()
cap.release()

# Get annotations
frame_data = get_object_positions_per_frame(annotation_path)
if frame_id not in frame_data:
    print(f'Frame {frame_id} not found')
    exit(1)

# Find a few objects that are clearly visible
objects_in_frame = frame_data[frame_id][:10]
print(f'\nObjects in frame {frame_id}:')
print('Annotation coordinates (world coords):')
for obj_id, x, y in objects_in_frame:
    print(f'  Object {obj_id}: ({x:.2f}, {y:.2f})')

print('\n=== Manual Calibration Instructions ===')
print('1. Look at the video frame and identify where objects actually appear')
print('2. Note the pixel coordinates where you see people/objects')
print('3. Match them with annotation coordinates above')
print()
print('For example, if Object 22 appears at pixel (500, 300) in the video,')
print('but annotation says (49.37, 28.83), we can calculate:')
print('  scale_x = (500 - offset_x) / 49.37')
print('  scale_y = (300 - offset_y) / 28.83')
print()
print('Alternatively, if you can identify 2+ reference points, we can solve:')
print('  pixel_x = annotation_x * scale_x + offset_x')
print('  pixel_y = annotation_y * scale_y + offset_y')
print()
print('Please provide:')
print('- Which objects are clearly visible in the video?')
print('- What are their pixel coordinates in the video?')
print('- Or describe roughly where objects appear (upper-left, center, etc.)')

# Create a simple visualization to help
h, w = frame.shape[:2]
vis_frame = frame.copy()

# Draw grid to help identify positions
grid_spacing = 100
for x in range(0, w, grid_spacing):
    cv2.line(vis_frame, (x, 0), (x, h), (100, 100, 100), 1)
for y in range(0, h, grid_spacing):
    cv2.line(vis_frame, (0, y), (w, y), (100, 100, 100), 1)

# Label corners
cv2.putText(vis_frame, '(0,0)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(vis_frame, f'({w},0)', (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(vis_frame, f'(0,{h})', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(vis_frame, f'({w},{h})', (w-150, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(vis_frame, f'Center: ({w//2},{h//2})', (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

cv2.imwrite('output/video_frame_with_grid.png', vis_frame)
print(f'\nSaved: output/video_frame_with_grid.png')
print('This shows the video frame with a grid to help identify pixel coordinates.')
print('Use this to find where objects actually appear!')

