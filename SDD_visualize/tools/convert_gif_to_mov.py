#!/usr/bin/env python3
import os
import sys
import imageio

input_gif = os.path.join('outputs', 'pedestrian_1_overlay.gif')
output_mov = os.path.join('outputs', 'pedestrian_1_overlay.mov')

if not os.path.exists(input_gif):
    print(f"Input GIF not found: {input_gif}")
    sys.exit(2)

reader = imageio.get_reader(input_gif)

# Try to infer fps from GIF metadata
try:
    meta = reader.get_meta_data()
    duration = meta.get('duration', None)
    if duration is None:
        fps = 10
    else:
        if duration <= 1:
            fps = 1.0 / float(duration) if duration > 0 else 10
        else:
            fps = 1000.0 / float(duration) if duration > 0 else 10
except Exception:
    fps = 10

print(f"Converting {input_gif} -> {output_mov} with fps={fps}")

# Use ffmpeg backend explicitly and set sensible codec/params for .mov
try:
    writer = imageio.get_writer(
        output_mov,
        format='ffmpeg',
        mode='I',
        fps=fps,
        codec='libx264',
        ffmpeg_log_level='error',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )
    for im in reader:
        writer.append_data(im)
    writer.close()
    print(f"Wrote {output_mov}")
except Exception as e:
    print("Failed to write MOV:", e)
    print("Ensure ffmpeg binary is installed on the system (e.g. 'brew install ffmpeg') and that the Python package imageio[ffmpeg] is available.")
    sys.exit(1)
