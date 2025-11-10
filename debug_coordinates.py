#!/usr/bin/env python3
"""Debug script to visualize where objects actually appear in the video."""

import cv2
import numpy as np
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.visualizer import calculate_coordinate_transform, transform_coordinates

# Load video frame
video_path = 'sdd_bookstore/bookstore_vid/video0/video.mp4'
annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
print(f'Video size: {w}x{h}')
print(f'Video center: ({w//2}, {h//2})')

# Get annotations
frame_data = get_object_positions_per_frame(annotation_path)
if 5000 in frame_data:
    # Create two test frames
    frame_raw = frame.copy()
    frame_transformed = frame.copy()
    
    # Calculate transformation
    coord_transform = calculate_coordinate_transform(annotation_path, w, h)
    scale_x, scale_y, offset_x, offset_y = coord_transform
    print(f'\nTransformation: scale=({scale_x:.2f}, {scale_y:.2f}), offset=({offset_x:.2f}, {offset_y:.2f})')
    
    # Draw objects at RAW coordinates (no transformation)
    print(f'\nDrawing objects at RAW annotation coordinates:')
    for obj_id, x, y in frame_data[5000][:15]:
        px, py = int(x), int(y)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(frame_raw, (px, py), 10, (0, 0, 255), 2)  # Red circles
            cv2.putText(frame_raw, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            print(f'  Object {obj_id}: annotation=({x:.1f}, {y:.1f}) -> pixel=({px}, {py}) [RAW]')
    
    # Draw objects at TRANSFORMED coordinates
    print(f'\nDrawing objects at TRANSFORMED coordinates:')
    for obj_id, x, y in frame_data[5000][:15]:
        px, py = transform_coordinates(x, y, scale_x, scale_y, offset_x, offset_y)
        px, py = int(px), int(py)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(frame_transformed, (px, py), 10, (0, 255, 0), 2)  # Green circles
            cv2.putText(frame_transformed, f'{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            dist_from_center = np.sqrt((px - w//2)**2 + (py - h//2)**2)
            print(f'  Object {obj_id}: annotation=({x:.1f}, {y:.1f}) -> pixel=({px}, {py}) [TRANSFORMED], dist={dist_from_center:.1f}px')
    
    # Draw video center marker
    cv2.circle(frame_raw, (w//2, h//2), 20, (255, 255, 0), 3)
    cv2.circle(frame_transformed, (w//2, h//2), 20, (255, 255, 0), 3)
    cv2.putText(frame_raw, 'CENTER', (w//2-30, h//2-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame_transformed, 'CENTER', (w//2-30, h//2-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Save frames
    cv2.imwrite('output/debug_raw_coordinates.png', frame_raw)
    cv2.imwrite('output/debug_transformed_coordinates.png', frame_transformed)
    print(f'\nSaved comparison frames:')
    print(f'  - output/debug_raw_coordinates.png (red circles = raw annotation coords)')
    print(f'  - output/debug_transformed_coordinates.png (green circles = transformed coords)')

