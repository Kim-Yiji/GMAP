#!/usr/bin/env python3
"""
Interactive calibration: Click on objects in the video to identify their positions.
This will calculate the correct transformation.
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

# Create annotation lookup
ann_coords = {obj_id: (x, y) for obj_id, x, y in frame_data[frame_id]}

# Store clicked points
clicked_points = []
current_obj_id = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_obj_id, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_obj_id is not None:
            # Save this point
            ann_x, ann_y = ann_coords[current_obj_id]
            clicked_points.append({
                'obj_id': current_obj_id,
                'pixel': (x, y),
                'annotation': (ann_x, ann_y)
            })
            print(f'✓ Object {current_obj_id}: pixel=({x}, {y}), annotation=({ann_x:.2f}, {ann_y:.2f})')
            
            # Draw marker
            cv2.circle(frame_copy, (x, y), 10, (0, 255, 0), 3)
            cv2.putText(frame_copy, f'{current_obj_id}', (x+15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration', frame_copy)
            
            # Ask for next object
            if len(clicked_points) < 3:
                print(f'\nClick on the next object (need {3 - len(clicked_points)} more)')
            else:
                print(f'\n✓ Got {len(clicked_points)} reference points. Calculating transformation...')
                calculate_transformation()

def calculate_transformation():
    """Calculate transformation from clicked points."""
    if len(clicked_points) < 2:
        print('Need at least 2 points')
        return
    
    # Extract points
    ann_points = np.array([p['annotation'] for p in clicked_points])
    pix_points = np.array([p['pixel'] for p in clicked_points])
    
    # Solve for scale and offset using least squares
    # pixel_x = annotation_x * scale_x + offset_x
    # pixel_y = annotation_y * scale_y + offset_y
    
    # For X
    A_x = np.column_stack([ann_points[:, 0], np.ones(len(ann_points))])
    scale_x, offset_x = np.linalg.lstsq(A_x, pix_points[:, 0], rcond=None)[0]
    
    # For Y
    A_y = np.column_stack([ann_points[:, 1], np.ones(len(ann_points))])
    scale_y, offset_y = np.linalg.lstsq(A_y, pix_points[:, 1], rcond=None)[0]
    
    print('\n' + '=' * 60)
    print('CALCULATED TRANSFORMATION:')
    print('=' * 60)
    print(f'Scale X: {scale_x:.4f}')
    print(f'Scale Y: {scale_y:.4f}')
    print(f'Offset X: {offset_x:.4f}')
    print(f'Offset Y: {offset_y:.4f}')
    print('\nVerification:')
    for p in clicked_points:
        obj_id = p['obj_id']
        ann_x, ann_y = p['annotation']
        pix_x, pix_y = p['pixel']
        calc_x = ann_x * scale_x + offset_x
        calc_y = ann_y * scale_y + offset_y
        error_x = abs(calc_x - pix_x)
        error_y = abs(calc_y - pix_y)
        print(f'  Object {obj_id}: expected=({pix_x}, {pix_y}), calculated=({calc_x:.1f}, {calc_y:.1f}), error=({error_x:.1f}, {error_y:.1f})')
    
    print('\n' + '=' * 60)
    print('UPDATE THE CODE WITH THESE VALUES:')
    print('=' * 60)
    print(f'scale_x = {scale_x:.4f}')
    print(f'scale_y = {scale_y:.4f}')
    print(f'offset_x = {offset_x:.4f}')
    print(f'offset_y = {offset_y:.4f}')
    print('=' * 60)

# Display frame
frame_copy = frame.copy()
h, w = frame.shape[:2]

# Draw grid
for x in range(0, w, 100):
    cv2.line(frame_copy, (x, 0), (x, h), (50, 50, 50), 1)
for y in range(0, h, 100):
    cv2.line(frame_copy, (0, y), (w, y), (50, 50, 50), 1)

# Show available objects
print('=' * 60)
print('INTERACTIVE CALIBRATION')
print('=' * 60)
print(f'\nVideo frame {frame_id}')
print(f'Video dimensions: {w}x{h}')
print(f'\nAvailable objects (annotation coordinates):')
for obj_id in sorted(ann_coords.keys())[:15]:
    x, y = ann_coords[obj_id]
    print(f'  Object {obj_id}: ({x:.2f}, {y:.2f})')

print('\n' + '=' * 60)
print('INSTRUCTIONS:')
print('=' * 60)
print('1. A window will open showing the video frame')
print('2. Enter an object ID (from the list above)')
print('3. Click on where that object appears in the video')
print('4. Repeat for at least 2-3 objects')
print('5. The transformation will be calculated automatically')
print('=' * 60)

cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

print('\nEnter object IDs one by one, then click on them in the video window.')
print('Type object ID and press Enter, then click on the object in the video.')
print('Type "done" when finished (need at least 2 points)')
print()

# Simple text-based interface
while len(clicked_points) < 10:
    obj_input = input(f'Enter object ID (or "done" to finish): ').strip()
    if obj_input.lower() == 'done':
        if len(clicked_points) >= 2:
            break
        else:
            print('Need at least 2 points!')
            continue
    
    try:
        obj_id = int(obj_input)
        if obj_id not in ann_coords:
            print(f'Object {obj_id} not found in this frame')
            continue
        
        current_obj_id = obj_id
        ann_x, ann_y = ann_coords[obj_id]
        print(f'Object {obj_id} selected: annotation=({ann_x:.2f}, {ann_y:.2f})')
        print('Now click on where this object appears in the video window...')
        
        cv2.imshow('Calibration', frame_copy)
        cv2.waitKey(0)
        
    except ValueError:
        print('Invalid input. Enter a number or "done"')

cv2.destroyAllWindows()

if len(clicked_points) >= 2:
    calculate_transformation()
else:
    print('Not enough points collected')

