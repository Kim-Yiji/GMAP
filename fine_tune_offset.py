#!/usr/bin/env python3
"""
Fine-tune offset interactively. Since movements match, scale is correct,
we just need to adjust the offset to align objects with people.
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

# Get annotations
frame_data = get_object_positions_per_frame(annotation_path)
if frame_id not in frame_data:
    print(f'Frame {frame_id} not found')
    exit(1)

# Calculate base transformation
coord_transform = calculate_coordinate_transform(annotation_path, frame.shape[1], frame.shape[0], preserve_position=True)
scale_x, scale_y, base_offset_x, base_offset_y = coord_transform

# Current offset adjustment
offset_adj_x = 0
offset_adj_y = 0

def draw_frame(offset_x_adj, offset_y_adj):
    """Draw frame with current offset adjustment."""
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Apply offset adjustment
    offset_x = base_offset_x + offset_x_adj
    offset_y = base_offset_y + offset_y_adj
    
    # Draw objects
    for obj_id, x, y in frame_data[frame_id][:20]:
        px = x * scale_x + offset_x
        py = y * scale_y + offset_y
        px, py = int(px), int(py)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(frame_copy, (px, py), 10, (0, 255, 0), 2)
            cv2.putText(frame_copy, f'id:{obj_id}', (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Show current offset
    cv2.putText(frame_copy, f'Offset adjustment: ({offset_x_adj:+d}, {offset_y_adj:+d})', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_copy, f'Total offset: ({offset_x:.1f}, {offset_y:.1f})', 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_copy, 'Arrow keys: adjust, Space: save, Esc: quit', 
               (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_copy, offset_x, offset_y

print('=' * 60)
print('FINE-TUNE OFFSET')
print('=' * 60)
print(f'Scale: ({scale_x:.2f}, {scale_y:.2f}) - CORRECT (movements match)')
print(f'Base offset: ({base_offset_x:.2f}, {base_offset_y:.2f})')
print(f'Current adjustment: ({offset_adj_x}, {offset_adj_y})')
print()
print('Use arrow keys to adjust offset:')
print('  Left/Right: adjust X offset')
print('  Up/Down: adjust Y offset')
print('  Space: save current offset')
print('  Esc: quit')
print('=' * 60)

cv2.namedWindow('Fine-tune Offset', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fine-tune Offset', 1200, 900)

step = 10  # Pixel step size

while True:
    display_frame, current_offset_x, current_offset_y = draw_frame(offset_adj_x, offset_adj_y)
    cv2.imshow('Fine-tune Offset', display_frame)
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == 27:  # Esc
        break
    elif key == 32:  # Space - save
        print(f'\n✓ Saved offset adjustment: ({offset_adj_x}, {offset_adj_y})')
        print(f'  Total offset: ({current_offset_x:.2f}, {current_offset_y:.2f})')
        print(f'\nUpdate the code with:')
        print(f'  manual_offset=({current_offset_x:.2f}, {current_offset_y:.2f})')
        
        # Save test image
        cv2.imwrite('output/final_offset_alignment.png', display_frame)
        print(f'  Saved: output/final_offset_alignment.png')
        break
    elif key == 81 or key == 2:  # Left arrow
        offset_adj_x -= step
        print(f'Offset X: {offset_adj_x} (left {step}px)')
    elif key == 83 or key == 3:  # Right arrow
        offset_adj_x += step
        print(f'Offset X: {offset_adj_x} (right {step}px)')
    elif key == 82 or key == 0:  # Up arrow
        offset_adj_y -= step
        print(f'Offset Y: {offset_adj_y} (up {step}px)')
    elif key == 84 or key == 1:  # Down arrow
        offset_adj_y += step
        print(f'Offset Y: {offset_adj_y} (down {step}px)')

cv2.destroyAllWindows()

