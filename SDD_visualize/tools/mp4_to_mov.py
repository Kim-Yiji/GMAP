#!/usr/bin/env python3
"""
Simple MP4 -> MOV conversion helper using ffmpeg.

Usage examples:
  python tools/mp4_to_mov.py                            # converts outputs/pedestrian_1_overlay.mp4 -> outputs/pedestrian_1_overlay.mov (stream-copy if possible)
  python tools/mp4_to_mov.py -i path/to/in.mp4 -o out.mov --reencode

Options:
  -i/--input     Input MP4 path (default: outputs/pedestrian_1_overlay.mp4)
  -o/--output    Output MOV path (default: same basename with .mov)
  --reencode     Re-encode video (libx264) and audio (aac) for maximum compatibility
  --crf          CRF value when reencoding (default 23)
  --preset       ffmpeg preset when reencoding (default medium)
  --overwrite    Overwrite output if exists

The script calls the system ffmpeg binary, so ensure ffmpeg is installed and on PATH (e.g. brew install ffmpeg).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print('Running:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('ffmpeg failed with exit code', e.returncode)
        sys.exit(e.returncode)


def main():
    p = argparse.ArgumentParser(description='Convert MP4 to MOV using ffmpeg')
    p.add_argument('-i', '--input', default='outputs/pedestrian_1_overlay.mp4', help='Input MP4 file')
    p.add_argument('-o', '--output', default=None, help='Output MOV file (defaults to input basename with .mov)')
    p.add_argument('--reencode', action='store_true', help='Re-encode video/audio for compatibility (libx264/aac)')
    p.add_argument('--crf', type=int, default=23, help='CRF for libx264 when reencoding (lower = higher quality)')
    p.add_argument('--preset', default='medium', help='ffmpeg preset for libx264')
    p.add_argument('--overwrite', action='store_true', help='Overwrite output if exists')

    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f'Input file not found: {inp}')
        sys.exit(2)

    out = Path(args.output) if args.output else inp.with_suffix('.mov')

    if out.exists() and not args.overwrite:
        print(f'Output already exists: {out} (use --overwrite to replace)')
        sys.exit(3)

    # Ensure ffmpeg is available
    if not shutil_which('ffmpeg'):
        print('ffmpeg not found on PATH. Install it (brew install ffmpeg)')
        sys.exit(4)

    if args.reencode:
        # re-encode with libx264 and aac
        cmd = [
            'ffmpeg',
            '-y' if args.overwrite else '-n',
            '-i', str(inp),
            '-c:v', 'libx264',
            '-crf', str(args.crf),
            '-preset', args.preset,
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            str(out)
        ]
    else:
        # try stream copy
        cmd = [
            'ffmpeg',
            '-y' if args.overwrite else '-n',
            '-i', str(inp),
            '-c', 'copy',
            str(out)
        ]

    run(cmd)
    print('Wrote', out)


def shutil_which(name):
    # small helper to avoid importing shutil at top-level for clarity
    from shutil import which
    return which(name)

if __name__ == '__main__':
    main()
