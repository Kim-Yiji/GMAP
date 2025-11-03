#!/usr/bin/env python3
"""
MOT (Multiple Object Tracking) 형식을 GMAP 형식으로 변환하는 스크립트

MOT 형식: frame_id track_id x1 y1 x2 y2 confidence class_id visibility "class_name"
GMAP 형식: frame_id ped_id x y

사용법:
python convert_mot_to_gmap.py --input_dir /raid/guest/newbie/yoonhee/DMRGCN/datasets/bookstore --output_dir ./copy_dmrgcn/datasets/bookstore
"""

import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def parse_mot_line(line):
    """MOT 형식의 한 줄을 파싱"""
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    
    frame_id = int(parts[0])
    track_id = int(parts[1])
    x1, y1, x2, y2 = map(float, parts[2:6])
    confidence = float(parts[6])
    class_id = int(parts[7])
    visibility = float(parts[8])
    
    # 바운딩 박스 중심점 계산
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    return frame_id, track_id, center_x, center_y, confidence, class_id, visibility


def convert_mot_to_gmap(input_file, output_file, min_confidence=0.5, min_visibility=0.5):
    """MOT 파일을 GMAP 형식으로 변환"""
    print(f"Converting {input_file} -> {output_file}")
    
    # 데이터를 저장할 딕셔너리: {frame_id: [(track_id, x, y), ...]}
    frame_data = {}
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip() == '':
                continue
                
            try:
                parsed = parse_mot_line(line)
                if parsed is None:
                    continue
                    
                frame_id, track_id, x, y, confidence, class_id, visibility = parsed
                
                # 필터링 조건
                if confidence < min_confidence or visibility < min_visibility:
                    continue
                
                if frame_id not in frame_data:
                    frame_data[frame_id] = []
                
                frame_data[frame_id].append((track_id, x, y))
                
            except Exception as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    # GMAP 형식으로 출력
    with open(output_file, 'w') as f:
        for frame_id in sorted(frame_data.keys()):
            for track_id, x, y in frame_data[frame_id]:
                f.write(f"{frame_id}\t{track_id}\t{x:.6f}\t{y:.6f}\n")
    
    print(f"Converted {len(frame_data)} frames with {sum(len(tracks) for tracks in frame_data.values())} total detections")


def process_dataset(input_dir, output_dir, dataset_name, split_ratios=(0.7, 0.15, 0.15)):
    """데이터셋을 처리하고 train/val/test로 분할"""
    print(f"Processing dataset: {dataset_name}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.join(output_dir, dataset_name, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, dataset_name, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, dataset_name, 'test'), exist_ok=True)
    
    # 모든 video 디렉토리 찾기
    video_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and item.startswith('video'):
            video_dirs.append(item)
    
    video_dirs.sort()
    print(f"Found {len(video_dirs)} video directories: {video_dirs}")
    
    # 분할 계산
    train_end = int(len(video_dirs) * split_ratios[0])
    val_end = train_end + int(len(video_dirs) * split_ratios[1])
    
    train_videos = video_dirs[:train_end]
    val_videos = video_dirs[train_end:val_end]
    test_videos = video_dirs[val_end:]
    
    print(f"Split: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # 각 split 처리
    for split_name, videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
        if not videos:
            continue
            
        print(f"\nProcessing {split_name} split...")
        split_data = []
        
        for video in videos:
            video_path = os.path.join(input_dir, video)
            annotations_file = os.path.join(video_path, 'annotations.txt')
            
            if not os.path.exists(annotations_file):
                print(f"Warning: {annotations_file} not found, skipping")
                continue
            
            # 임시 파일로 변환
            temp_file = f"/tmp/{dataset_name}_{video}_temp.txt"
            convert_mot_to_gmap(annotations_file, temp_file)
            
            # 데이터 로드
            with open(temp_file, 'r') as f:
                for line in f:
                    split_data.append(line.strip())
            
            # 임시 파일 삭제
            os.remove(temp_file)
        
        # 파일로 저장
        output_file = os.path.join(output_dir, dataset_name, split_name, f"{dataset_name}_{split_name}.txt")
        with open(output_file, 'w') as f:
            for line in split_data:
                f.write(line + '\n')
        
        print(f"Saved {len(split_data)} lines to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert MOT format to GMAP format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing MOT datasets')
    parser.add_argument('--output_dir', default='./copy_dmrgcn/datasets', help='Output directory for GMAP datasets')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to convert (default: all)')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence threshold')
    parser.add_argument('--min_visibility', type=float, default=0.5, help='Minimum visibility threshold')
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터셋 목록 확인
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [d for d in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, d))]
    
    print(f"Converting datasets: {datasets}")
    
    # 각 데이터셋 처리
    for dataset in datasets:
        input_path = os.path.join(args.input_dir, dataset)
        if not os.path.isdir(input_path):
            print(f"Warning: {input_path} is not a directory, skipping")
            continue
        
        try:
            process_dataset(input_path, args.output_dir, dataset)
            print(f"✅ Successfully converted {dataset}")
        except Exception as e:
            print(f"❌ Error converting {dataset}: {e}")
    
    print("\n🎉 Conversion completed!")


if __name__ == '__main__':
    main()




