"""
전처리 스크립트: SDD Coupa 데이터셋을 전처리하여 .pt 파일로 저장

dataloader_test.py의 ThreatTrajectoryDataset을 사용하여 
3개 relation (A_dist, A_disp, A_threat) 그래프를 생성하고 저장합니다.
"""

import os
import sys
import math
import torch
import numpy as np
from datetime import datetime
import time
from pathlib import Path

# DMRGCN 모듈 import (청킹 버전은 modeling/DMRGCN 하위의 ThreatTrajectoryDataset 사용)
sys.path.insert(0, '/raid/guest/OATMeal_Queens/cy_2nd_attempt/modeling/DMRGCN')
from utils.dataloader_test import ThreatTrajectoryDataset

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "200"))

def log(message, level="INFO"):
    """로그 메시지 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def log_section(title):
    """섹션 구분선 출력"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def _save_chunk(dataset, split_dir, base_name, chunk_idx, seq_start_idx, seq_end_idx):
    """주어진 시퀀스 구간을 잘라 별도 .pt 파일로 저장"""
    ped_start = dataset.seq_start_end[seq_start_idx][0]
    ped_end = dataset.seq_start_end[seq_end_idx - 1][1]

    chunk_seq = dataset.seq_start_end[seq_start_idx:seq_end_idx]
    seq_start_end = [(s - ped_start, e - ped_start) for (s, e) in chunk_seq]

    cache = {
        'obs_traj': dataset.obs_traj[ped_start:ped_end],
        'pred_traj': dataset.pred_traj[ped_start:ped_end],
        'seq_start_end': seq_start_end,
        'V_obs': dataset.V_obs[seq_start_idx:seq_end_idx],
        'A_obs': dataset.A_obs[seq_start_idx:seq_end_idx],
        'V_pred': dataset.V_pred[seq_start_idx:seq_end_idx],
        'ped_masks': dataset.ped_masks[seq_start_idx:seq_end_idx],
    }

    chunk_name = f"{base_name}_part{chunk_idx:03d}.pt"
    chunk_path = os.path.join(split_dir, chunk_name)

    torch.save(cache, chunk_path)
    size_mb = os.path.getsize(chunk_path) / (1024 ** 2)
    log(f"    └ 저장 완료: {chunk_name} ({size_mb:.2f} MB)")

    return chunk_path, size_mb


def preprocess_and_save(data_path, output_prefix, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=0,
                        chunk_size=200):
    """
    데이터를 전처리하고 .pt 파일로 저장
    
    Args:
        data_dir: 전처리할 데이터 디렉토리 (.txt 파일들이 있는 곳)
        output_path: 저장할 .pt 파일 경로
        obs_len: 관찰 길이
        pred_len: 예측 길이
        skip: 프레임 스킵
        threshold: 비선형 궤적 임계값
        min_ped: 최소 보행자 수
    """
    source = "directory" if os.path.isdir(data_path) else "file"
    log(f"전처리 시작: {data_path} ({source})")
    log(f"출력 prefix: {output_prefix}")
    log(f"설정: obs_len={obs_len}, pred_len={pred_len}, skip={skip}, threshold={threshold}, min_ped={min_ped}")
    
    # 입력 파일 확인
    if os.path.isdir(data_path):
        txt_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
        txt_files = sorted(txt_files)
    elif data_path.endswith('.txt'):
        txt_files = [data_path]
    else:
        txt_files = []

    log(f"입력 .txt 파일 수: {len(txt_files)}")
    
    if not txt_files:
        log("전처리할 .txt 파일이 없습니다!", "ERROR")
        return []
    
    start_time = time.time()
    
    log("ThreatTrajectoryDataset 초기화 중...")
    log("  → 데이터 로드 및 궤적 추출 중...")
    log("  → 3개 relation 그래프 생성 중 (A_dist, A_disp, A_threat)...")
    
    # ThreatTrajectoryDataset으로 데이터 로드 및 전처리
    try:
        dataset = ThreatTrajectoryDataset(
            data_dir=txt_files[0] if len(txt_files) == 1 else data_path,
            obs_len=obs_len,
            pred_len=pred_len,
            skip=skip,
            threshold=threshold,
            min_ped=min_ped,
            delim='\t'
        )
    except Exception as e:
        log(f"전처리 중 오류 발생: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False
    
    elapsed_time = time.time() - start_time
    
    log(f"\n전처리 완료! (소요 시간: {elapsed_time:.2f}초)")
    log(f"  - 시퀀스 수: {dataset.num_seq:,}")
    log(f"  - 총 보행자 궤적 수: {len(dataset.obs_traj):,}")
    log(f"  - 최대 노드 수 (pedestrians + obstacles): {dataset.max_nodes}")
    log(f"  - 관찰 궤적 shape: {dataset.obs_traj.shape}")
    log(f"  - 예측 궤적 shape: {dataset.pred_traj.shape}")
    
    # 그래프 통계
    log(f"  - 그래프 수 (V_obs, A_obs): {len(dataset.V_obs)}")
    if len(dataset.V_obs) > 0:
        log(f"  - H_obs shape 예시: {dataset.V_obs[0].shape}")
        log(f"  - A_obs shape 예시: {dataset.A_obs[0].shape} (3 relations: [A_dist, A_disp, A_threat])")
        if dataset.A_obs[0].shape[0] == 3:
            log(f"    ✓ 3개 relation 확인됨!")
        else:
            log(f"    ✗ Warning: Expected 3 relations, got {dataset.A_obs[0].shape[0]}", "WARNING")
    
    # 데이터를 딕셔너리로 저장
    log(f"\n데이터 딕셔너리 구성 중...")
    num_chunks = math.ceil(dataset.num_seq / chunk_size)
    log(f"\n총 {dataset.num_seq}개 시퀀스를 {num_chunks}개 파일로 분할 저장합니다. (chunk_size={chunk_size})")

    chunk_paths = []
    split_dir = os.path.dirname(output_prefix)
    os.makedirs(split_dir, exist_ok=True)

    for chunk_idx in range(num_chunks):
        seq_start_idx = chunk_idx * chunk_size
        seq_end_idx = min(dataset.num_seq, (chunk_idx + 1) * chunk_size)
        log(f"  > Chunk {chunk_idx+1}/{num_chunks}: seq[{seq_start_idx}:{seq_end_idx})")
        chunk_path, size_mb = _save_chunk(dataset, split_dir, os.path.basename(output_prefix), chunk_idx,
                                          seq_start_idx, seq_end_idx)
        chunk_paths.append((chunk_path, size_mb))

    return chunk_paths

def main():
    """메인 함수: train, val, test 각각 전처리"""
    log_section("SDD Coupa 데이터셋 전처리 시작")
    
    base_dir = "/raid/guest/OATMeal_Queens/cy_2nd_attempt/modeling/DMRGCN/SDD_datasets/sdd_coupa"
    
    log(f"기본 데이터 디렉토리: {base_dir}")
    
    # 각 split에 대해 처리
    splits = ['train', 'val', 'test']
    
    overall_start_time = time.time()
    
    for split_idx, split in enumerate(splits, 1):
        log_section(f"[{split_idx}/{len(splits)}] {split.upper()} Split 처리")
        
        split_start_time = time.time()
        
        data_dir = os.path.join(base_dir, split)
        output_prefix = os.path.join(data_dir, f'sdd_coupa_{split}_threat')
        
        # 디렉토리 존재 확인
        if not os.path.exists(data_dir):
            log(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}", "ERROR")
            continue
        
        txt_files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith('.txt')
        )
        if not txt_files:
            log(f"{split} split에서 .txt를 찾을 수 없습니다.", "WARNING")
            continue

        log(f"{split} split 영상 수: {len(txt_files)}")

        generated_chunks = []

        for vid_idx, txt_fname in enumerate(txt_files, 1):
            video_path = os.path.join(data_dir, txt_fname)
            video_stem = Path(txt_fname).stem
            video_prefix = os.path.join(data_dir, f'sdd_coupa_{split}_{video_stem}')

            log_section(f"{split.upper()} :: {video_stem} ({vid_idx}/{len(txt_files)})")
            log(f"입력 파일: {video_path}")
            log(f"출력 파일 prefix: {video_prefix}_partXXX.pt")

            try:
                chunk_paths = preprocess_and_save(
                    video_path,
                    video_prefix,
                    min_ped=0,
                    chunk_size=CHUNK_SIZE
                )
                if chunk_paths:
                    generated_chunks.extend(chunk_paths)
            except Exception as e:
                log(f"{video_stem} 전처리 중 오류: {str(e)}", "ERROR")
                import traceback
                traceback.print_exc()

        split_elapsed = time.time() - split_start_time
        log(f"\n{split.upper()} Split 완료! (총 소요 시간: {split_elapsed:.2f}초)", "SUCCESS")
    
    overall_elapsed = time.time() - overall_start_time
    
    log_section("전처리 완료 요약")
    log(f"총 소요 시간: {overall_elapsed:.2f}초 ({overall_elapsed/60:.2f}분)")
    log(f"\n생성된 파일:")
    
    total_size = 0
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        for root, _, files in os.walk(split_dir):
            for fname in sorted(f for f in files if f.endswith('.pt')):
                path = os.path.join(root, fname)
                size_mb = os.path.getsize(path) / (1024**2)
                total_size += size_mb
                log(f"  ✓ {path}")
                log(f"    크기: {size_mb:.2f} MB")
    
    log(f"\n총 저장된 데이터 크기: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    log_section("모든 작업 완료!")

if __name__ == "__main__":
    main()

