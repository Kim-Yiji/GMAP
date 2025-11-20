#!/usr/bin/env python3
"""
MOT 형식에서 Pedestrian만 필터링해서 GMAP 형식으로 변환하는 스크립트
"""

import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random


# Stanford Drone Dataset 영상 길이 정보 (초 단위)
VIDEO_DURATIONS = {
    'bookstore': {'video0': 444, 'video1': 485, 'video2': 485, 'video3': 485, 'video4': 485, 'video5': 485, 'video6': 485},
    'coupa': {'video0': 399, 'video1': 399, 'video2': 399, 'video3': 399},
    'deathCircle': {'video0': 424, 'video1': 469, 'video2': 14, 'video3': 416, 'video4': 15},
    'gates': {'video0': 300, 'video1': 300, 'video2': 300, 'video3': 306, 'video4': 73, 'video5': 69, 'video6': 69, 'video7': 73, 'video8': 73},
    'hyang': {'video0': 379, 'video1': 448, 'video2': 409, 'video3': 409, 'video4': 268, 'video5': 355, 'video6': 331, 'video7': 19, 'video8': 19, 'video9': 19, 'video10': 331, 'video11': 331, 'video12': 331, 'video13': 331, 'video14': 331},
    'little': {'video0': 50, 'video1': 469, 'video2': 469, 'video3': 469},
    'nexus': {'video0': 423, 'video1': 423, 'video2': 382, 'video3': 382, 'video4': 423, 'video5': 35, 'video6': 35, 'video7': 35, 'video8': 400, 'video9': 400, 'video10': 400, 'video11': 382},
    'quad': {'video0': 16, 'video1': 16, 'video2': 16, 'video3': 16}
}


def filter_videos_by_duration(video_dirs, dataset_name, min_duration=90):
    """영상 길이를 기준으로 비디오 필터링"""
    if dataset_name not in VIDEO_DURATIONS:
        print(f"Warning: No duration info for dataset '{dataset_name}', using all videos")
        return video_dirs, []
    
    durations = VIDEO_DURATIONS[dataset_name]
    valid_videos = []
    filtered_videos = []
    
    for video in video_dirs:
        if video in durations:
            duration = durations[video]
            if duration >= min_duration:
                valid_videos.append(video)
            else:
                filtered_videos.append((video, duration))
        else:
            # 길이 정보가 없으면 포함
            valid_videos.append(video)
    
    if filtered_videos:
        print(f"  Filtered out {len(filtered_videos)} short videos:")
        for video, duration in filtered_videos:
            print(f"    - {video}: {duration}초 ({duration//60}:{duration%60:02d})")
    
    return valid_videos, filtered_videos


def balanced_split_by_duration(video_dirs, dataset_name, split_ratios=(0.7, 0.15, 0.15)):
    """영상 길이를 고려한 균등 분할"""
    if dataset_name not in VIDEO_DURATIONS:
        # 길이 정보가 없으면 기존 방식 사용
        train_end = int(len(video_dirs) * split_ratios[0])
        val_end = train_end + int(len(video_dirs) * split_ratios[1])
        return video_dirs[:train_end], video_dirs[train_end:val_end], video_dirs[val_end:]
    
    durations = VIDEO_DURATIONS[dataset_name]
    
    # 길이별로 정렬 (긴 것부터)
    video_with_durations = [(v, durations.get(v, 0)) for v in video_dirs]
    video_with_durations.sort(key=lambda x: x[1], reverse=True)
    
    # 라운드로빈 방식으로 균등 분배
    train_videos = []
    val_videos = []
    test_videos = []
    
    for i, (video, duration) in enumerate(video_with_durations):
        if i % 3 == 0:
            train_videos.append(video)
        elif i % 3 == 1:
            val_videos.append(video)
        else:
            test_videos.append(video)
    
    # 최소 개수 보장 (val이 없으면 train에서 이동)
    if not val_videos and len(train_videos) > 1:
        val_videos.append(train_videos.pop())
    
    # 최소 개수 보장 (test가 없으면 train에서 이동)
    if not test_videos and len(train_videos) > 1:
        test_videos.append(train_videos.pop())
    
    return train_videos, val_videos, test_videos


def parse_mot_line(line, scale_factor=100.0):
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
    class_name = parts[9].strip('"')  # 따옴표 제거
    
    # 바운딩 박스 중심점 계산 및 좌표 정규화
    center_x = (x1 + x2) / 2.0 / scale_factor  # 스케일 팩터로 나누어 ETH 스케일에 맞춤
    center_y = (y1 + y2) / 2.0 / scale_factor
    
    return frame_id, track_id, center_x, center_y, confidence, class_id, visibility, class_name


def convert_mot_to_gmap(input_file, output_file, min_confidence=0.5, min_visibility=0.5, scale_factor=100.0, target_classes=None):
    """MOT 파일을 GMAP 형식으로 변환"""
    if target_classes is None:
        target_classes = ["Pedestrian"]  # 기본값: Pedestrian만
    
    class_filter_desc = "All classes" if len(target_classes) > 1 else f"{target_classes[0]} only"
    print(f"Converting {input_file} -> {output_file} ({class_filter_desc}, scale_factor={scale_factor})")
    
    # 데이터를 저장할 딕셔너리: {frame_id: [(track_id, x, y), ...]}
    frame_data = {}
    target_count = 0
    total_count = 0
    class_counts = {}
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip() == '':
                continue
                
            try:
                parsed = parse_mot_line(line, scale_factor)
                if parsed is None:
                    continue
                    
                frame_id, track_id, x, y, confidence, class_id, visibility, class_name = parsed
                total_count += 1
                
                # 클래스 통계
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                # 타겟 클래스 필터링
                if class_name not in target_classes:
                    continue
                
                # 필터링 조건
                if confidence < min_confidence or visibility < min_visibility:
                    continue
                
                target_count += 1
                
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
    
    print(f"Converted {len(frame_data)} frames with {sum(len(tracks) for tracks in frame_data.values())} target detections")
    print(f"Target classes ratio: {target_count}/{total_count} ({target_count/total_count*100:.1f}%)")
    
    # 클래스별 통계 출력
    if len(class_counts) > 1:
        print(f"Class distribution: {dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))}")


def process_dataset(input_dir, output_dir, dataset_name, split_ratios=(0.7, 0.15, 0.15), scale_factor=100.0, min_duration=90, exclude_datasets=None, target_classes=None):
    """데이터셋을 처리하고 train/val/test로 분할"""
    if target_classes is None:
        target_classes = ["Pedestrian"]  # 기본값: Pedestrian만
    
    class_desc = "All classes" if len(target_classes) > 1 else f"{target_classes[0]} only"
    print(f"Processing dataset: {dataset_name} ({class_desc}, scale_factor={scale_factor}, min_duration={min_duration}s)")
    
    # 제외할 데이터셋 체크
    if exclude_datasets and dataset_name in exclude_datasets:
        print(f"  ⚠️ Dataset '{dataset_name}' is in exclude list, skipping...")
        return
    
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
    
    # 짧은 영상 필터링
    valid_videos, filtered_videos = filter_videos_by_duration(video_dirs, dataset_name, min_duration)
    
    if not valid_videos:
        print(f"  ❌ No valid videos after filtering (all videos < {min_duration}s), skipping dataset")
        return
    
    print(f"Valid videos after filtering: {len(valid_videos)}/{len(video_dirs)} ({len(valid_videos)/len(video_dirs)*100:.1f}%)")
    
    # 개선된 분할 방식 적용
    train_videos, val_videos, test_videos = balanced_split_by_duration(valid_videos, dataset_name, split_ratios)
    
    print(f"Balanced split: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # 각 split의 평균 길이 출력
    if dataset_name in VIDEO_DURATIONS:
        durations = VIDEO_DURATIONS[dataset_name]
        if train_videos:
            train_avg = sum(durations.get(v, 0) for v in train_videos) / len(train_videos)
            print(f"  Train avg duration: {train_avg/60:.1f}min")
        if val_videos:
            val_avg = sum(durations.get(v, 0) for v in val_videos) / len(val_videos)
            print(f"  Val avg duration: {val_avg/60:.1f}min")
        if test_videos:
            test_avg = sum(durations.get(v, 0) for v in test_videos) / len(test_videos)
            print(f"  Test avg duration: {test_avg/60:.1f}min")
    
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
            class_suffix = "all" if len(target_classes) > 1 else target_classes[0].lower()
            temp_file = f"/tmp/{dataset_name}_{video}_{class_suffix}_temp.txt"
            convert_mot_to_gmap(annotations_file, temp_file, scale_factor=scale_factor, target_classes=target_classes)
            
            # 데이터 로드
            with open(temp_file, 'r') as f:
                for line in f:
                    split_data.append(line.strip())
            
            # 임시 파일 삭제
            os.remove(temp_file)
        
        # 파일로 저장
        class_suffix = "all" if len(target_classes) > 1 else target_classes[0].lower()
        output_file = os.path.join(output_dir, dataset_name, split_name, f"{dataset_name}_{split_name}_{class_suffix}.txt")
        with open(output_file, 'w') as f:
            for line in split_data:
                f.write(line + '\n')
        
        print(f"Saved {len(split_data)} lines to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert MOT format to GMAP format with flexible class filtering')
    parser.add_argument('--input_dir', required=True, help='Input directory containing MOT datasets')
    parser.add_argument('--output_dir', default='./datasets', help='Output directory for GMAP datasets')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to convert (default: all)')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence threshold')
    parser.add_argument('--min_visibility', type=float, default=0.5, help='Minimum visibility threshold')
    parser.add_argument('--scale_factor', type=float, default=100.0, help='Scaling factor for coordinate normalization (default: 100.0 for ETH, 329.0 for Stanford)')
    parser.add_argument('--min_duration', type=int, default=90, help='Minimum video duration in seconds (default: 90s)')
    parser.add_argument('--exclude_datasets', nargs='+', default=['quad'], help='Datasets to exclude (default: quad)')
    parser.add_argument('--balanced_split', action='store_true', default=True, help='Use balanced split based on video duration')
    
    # 클래스 필터링 옵션
    parser.add_argument('--target_classes', nargs='+', default=['Pedestrian'], 
                       choices=['Pedestrian', 'Biker', 'Car', 'Bus', 'Cart', 'Skater'],
                       help='Target classes to include (default: Pedestrian only)')
    parser.add_argument('--all_classes', action='store_true', 
                       help='Include all classes (overrides --target_classes)')
    
    # 실험 모드
    parser.add_argument('--experiment_mode', choices=['pedestrian_only', 'all_classes', 'both'], default='pedestrian_only',
                       help='Experiment mode: pedestrian_only, all_classes, or both (default: pedestrian_only)')
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 클래스 설정
    if args.all_classes:
        target_classes = ['Pedestrian', 'Biker', 'Car', 'Bus', 'Cart', 'Skater']
    else:
        target_classes = args.target_classes
    
    # 실험 모드별 설정
    experiments = []
    if args.experiment_mode == 'pedestrian_only':
        experiments = [(['Pedestrian'], 'pedestrian')]
    elif args.experiment_mode == 'all_classes':
        experiments = [(['Pedestrian', 'Biker', 'Car', 'Bus', 'Cart', 'Skater'], 'all')]
    elif args.experiment_mode == 'both':
        experiments = [
            (['Pedestrian'], 'pedestrian'),
            (['Pedestrian', 'Biker', 'Car', 'Bus', 'Cart', 'Skater'], 'all')
        ]
    else:
        experiments = [(target_classes, 'custom')]
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터셋 목록 확인
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [d for d in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, d))]
    
    # 제외할 데이터셋 필터링
    if args.exclude_datasets:
        original_count = len(datasets)
        datasets = [d for d in datasets if d not in args.exclude_datasets]
        if len(datasets) < original_count:
            excluded = [d for d in args.exclude_datasets if d in [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]]
            print(f"Excluding datasets: {excluded}")
    
    print(f"Experiment mode: {args.experiment_mode}")
    print(f"Processing datasets: {datasets}")
    print(f"Settings: scale_factor={args.scale_factor}, min_duration={args.min_duration}s")
    
    # 각 실험별로 처리
    all_results = {}
    
    for exp_classes, exp_suffix in experiments:
        print(f"\n{'='*60}")
        class_desc = "All classes" if len(exp_classes) > 1 else f"{exp_classes[0]} only"
        print(f"🔬 EXPERIMENT: {class_desc}")
        print(f"{'='*60}")
        
        # 실험별 출력 디렉토리
        exp_output_dir = os.path.join(args.output_dir, f"stanford_{exp_suffix}")
        os.makedirs(exp_output_dir, exist_ok=True)
        
        successful_conversions = []
        failed_conversions = []
        
        for dataset in datasets:
            input_path = os.path.join(args.input_dir, dataset)
            if not os.path.isdir(input_path):
                print(f"Warning: {input_path} is not a directory, skipping")
                continue
            
            try:
                process_dataset(
                    input_path, 
                    exp_output_dir, 
                    dataset, 
                    scale_factor=args.scale_factor,
                    min_duration=args.min_duration,
                    exclude_datasets=args.exclude_datasets,
                    target_classes=exp_classes
                )
                successful_conversions.append(dataset)
                print(f"✅ Successfully converted {dataset} ({class_desc})")
            except Exception as e:
                failed_conversions.append((dataset, str(e)))
                print(f"❌ Error converting {dataset}: {e}")
        
        all_results[exp_suffix] = {
            'successful': successful_conversions,
            'failed': failed_conversions,
            'classes': exp_classes
        }
    
    # 전체 결과 요약
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    
    for exp_suffix, results in all_results.items():
        class_desc = "All classes" if len(results['classes']) > 1 else f"{results['classes'][0]} only"
        print(f"\n📊 {class_desc.upper()} EXPERIMENT:")
        print(f"✅ Successful: {len(results['successful'])} datasets")
        if results['successful']:
            print(f"   {results['successful']}")
        
        if results['failed']:
            print(f"❌ Failed: {len(results['failed'])} datasets")
            for dataset, error in results['failed']:
                print(f"   {dataset}: {error}")
    
    if args.exclude_datasets:
        print(f"\n⚠️ Excluded datasets: {args.exclude_datasets}")


if __name__ == '__main__':
    main()
