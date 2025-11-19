#!/usr/bin/env python3
import os
import sys
import imageio

# 입력 GIF (기본값)
input_gif = os.path.join('outputs', 'pedestrian_1_overlay.gif')
# 출력 AVI
output_avi = os.path.join('outputs', 'pedestrian_1_overlay.avi')

if not os.path.exists(input_gif):
    print(f"Input GIF not found: {input_gif}")
    sys.exit(2)

reader = imageio.get_reader(input_gif)
# GIF 메타데이터에서 duration(ms) 가져오기 시도
try:
    meta = reader.get_meta_data()
    # 일부 imageio 버전은 duration을 초 단위로 제공할 수 있으니 방어적으로 처리
    duration = meta.get('duration', None)
    if duration is None:
        # duration이 없으면 기본 fps 사용
        fps = 10
    else:
        # duration이 밀리초 단위일 때가 많음
        # 만약 duration이 0~1 사이(초 단위)라면 초->밀리초 변환 고려
        if duration <= 1:
            # 초 단위로 추정
            fps = 1.0 / float(duration) if duration > 0 else 10
        else:
            # 밀리초 단위로 추정
            fps = 1000.0 / float(duration) if duration > 0 else 10
except Exception:
    fps = 10

print(f"Converting {input_gif} -> {output_avi} with fps={fps}")

# AVI로 쓰기 (ffmpeg backend 필요). format/FFMPEG을 명시해 ffmpeg 플러그인을 사용하도록 시도합니다.
try:
    # imageio가 올바른 플러그인을 선택하지 못할 수 있어서 format='FFMPEG' 명시
    writer = imageio.get_writer(output_avi, format='FFMPEG', fps=fps, codec='mpeg4')
    for im in reader:
        writer.append_data(im)
    writer.close()
    print(f"Wrote {output_avi}")
except Exception as e:
    print("Failed to write AVI:", e)
    print("If this fails, please ensure ffmpeg binary is available or install system ffmpeg (e.g. 'brew install ffmpeg').")
    sys.exit(1)
