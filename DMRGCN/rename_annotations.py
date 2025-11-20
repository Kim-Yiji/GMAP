#!/usr/bin/env python3
import os
from pathlib import Path

base_dir = Path('/raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets')

# 모든 annotations.txt 파일 찾기
annotations_files = sorted(list(base_dir.rglob('annotations.txt')))

print(f"Found {len(annotations_files)} annotations.txt files\n")

# 각 파일 이름 변경
renamed_count = 0
skipped_count = 0
error_count = 0

for ann_file in annotations_files:
    # 경로에서 데이터셋명과 video 번호 추출
    try:
        rel_path = ann_file.relative_to(base_dir)
        path_parts = rel_path.parts
        
        # annotations.txt는 video 폴더 안에 있으므로
        # path_parts = (dataset, video, 'annotations.txt')
        if len(path_parts) >= 3 and path_parts[-1] == 'annotations.txt':
            dataset_name = path_parts[0]
            video_name = path_parts[1]
            
            # 새 파일명 생성: {dataset}_video{num}.txt
            new_name = f"{dataset_name}_{video_name}.txt"
            new_path = ann_file.parent / new_name
            
            # 파일 이름 변경
            if not new_path.exists():
                ann_file.rename(new_path)
                print(f"✓ Renamed: {dataset_name}/{video_name}/annotations.txt -> {new_name}")
                renamed_count += 1
            else:
                print(f"⚠ Warning: {new_path.name} already exists, skipping {ann_file}")
                skipped_count += 1
        else:
            print(f"✗ Error: Unexpected path structure: {ann_file}")
            error_count += 1
    except Exception as e:
        print(f"✗ Error processing {ann_file}: {e}")
        error_count += 1

print(f"\n{'='*60}")
print(f"Total renamed: {renamed_count} files")
print(f"Skipped: {skipped_count} files")
print(f"Errors: {error_count} files")
print(f"{'='*60}")

