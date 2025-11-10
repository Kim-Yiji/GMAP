#!/usr/bin/env python3
"""
Interactive tool to manually identify where objects appear in the video.
This will help us calculate the correct transformation.
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

# Create an annotated frame showing object IDs at annotation coordinates
# This will help identify which objects are which
h, w = frame.shape[:2]
annotated_frame = frame.copy()

# Draw grid for reference
for x in range(0, w, 200):
    cv2.line(annotated_frame, (x, 0), (x, h), (50, 50, 50), 1)
    cv2.putText(annotated_frame, str(x), (x+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
for y in range(0, h, 200):
    cv2.line(annotated_frame, (0, y), (w, y), (50, 50, 50), 1)
    cv2.putText(annotated_frame, str(y), (5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Show center
cv2.circle(annotated_frame, (w//2, h//2), 10, (0, 255, 255), 2)
cv2.putText(annotated_frame, 'CENTER', (w//2-40, h//2-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv2.imwrite('output/video_frame_for_identification.png', annotated_frame)

print('=' * 60)
print('MANUAL IDENTIFICATION REQUIRED')
print('=' * 60)
print(f'\nVideo frame saved: output/video_frame_for_identification.png')
print(f'Frame ID: {frame_id}')
print(f'Video dimensions: {w}x{h}')
print('\nObjects in this frame (annotation coordinates):')
for obj_id, x, y in frame_data[frame_id][:20]:
    print(f'  Object {obj_id}: annotation=({x:.2f}, {y:.2f})')

print('\n' + '=' * 60)
print('INSTRUCTIONS:')
print('=' * 60)
print('1. Open output/video_frame_for_identification.png')
print('2. Look at the video frame and identify where PEOPLE/OBJECTS actually appear')
print('3. For at least 2-3 objects that are clearly visible, provide:')
print('   - Object ID (from the list above)')
print('   - Pixel coordinates (x, y) where that object appears in the video')
print('   - Use the grid to estimate coordinates')
print('\nExample:')
print('  "Object 22 appears at pixel (800, 600)"')
print('  "Object 23 appears at pixel (950, 580)"')
print('\nOnce we have 2+ reference points, we can calculate the correct')
print('transformation: pixel = annotation * scale + offset')
print('=' * 60)

