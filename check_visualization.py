#!/usr/bin/env python3
"""Check what's in the visualization output."""

import cv2
import json
import numpy as np

# Load a frame from the video
cap = cv2.VideoCapture('output/bookstore_video0_graph_full.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
cap.release()

print(f'Frame shape: {frame.shape}')
print(f'Frame loaded: {ret}')

# Load metadata
with open('output/bookstore_video0_graph_full_metadata.json', 'r') as f:
    data = json.load(f)

f0 = data['frames'][0]
print(f'\nFrame 0 (frame_id={f0["frame_id"]}):')
print(f'  Target ID: {f0["target_id"]}')
print(f'  Target position: {f0["target_position"]}')
print(f'  Number of interactions: {len(f0["interactions"])}')

if f0['interactions']:
    print(f'\n  First 5 obstacles:')
    for i, obs in enumerate(f0['interactions'][:5], 1):
        print(f'    {i}. ID={obs["object_id"]}, pos={obs["position"]}, score={obs["score"]:.3f}')
        # Check if position is within frame bounds
        x, y = obs["position"]
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            print(f'       -> Position is within frame bounds')
        else:
            print(f'       -> WARNING: Position is OUT OF BOUNDS!')

# Check target position
target_x, target_y = f0["target_position"]
print(f'\nTarget position check:')
print(f'  Target: ({target_x}, {target_y})')
print(f'  Frame size: {frame.shape[1]}x{frame.shape[0]}')
if 0 <= target_x < frame.shape[1] and 0 <= target_y < frame.shape[0]:
    print(f'  -> Target position is within frame bounds')
else:
    print(f'  -> WARNING: Target position is OUT OF BOUNDS!')

# Save a sample frame for inspection
cv2.imwrite('output/debug_frame.png', frame)
print(f'\nSaved debug frame to output/debug_frame.png')

