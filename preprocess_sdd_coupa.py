"""
전처리 스크립트: SDD Coupa 데이터셋을 전처리하여 .pt 파일로 저장

dataloader_test.py의 ThreatTrajectoryDataset을 사용하여 
3개 relation (A_dist, A_disp, A_threat) 그래프를 생성하고 저장합니다.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import time

# DMRGCN 모듈 import
sys.path.insert(0, '/raid/guest/OATMeal_Queens/cy_2nd_attempt/DMRGCN')
from utils.dataloader_test import ThreatTrajectoryDataset

def log(message, level="INFO"):
    """로그 메시지 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def log_section(title):
    """섹션 구분선 출력"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def preprocess_and_save(data_dir, output_path, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=0):
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
    log(f"전처리 시작: {data_dir}")
    log(f"출력 경로: {output_path}")
    log(f"설정: obs_len={obs_len}, pred_len={pred_len}, skip={skip}, threshold={threshold}, min_ped={min_ped}")
    
    # 입력 파일 확인
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    log(f"입력 .txt 파일 수: {len(txt_files)}")
    
    if len(txt_files) == 0:
        log("전처리할 .txt 파일이 없습니다!", "ERROR")
        return False
    
    start_time = time.time()
    
    log("ThreatTrajectoryDataset 초기화 중...")
    log("  → 데이터 로드 및 궤적 추출 중...")
    log("  → 3개 relation 그래프 생성 중 (A_dist, A_disp, A_threat)...")
    
    # ThreatTrajectoryDataset으로 데이터 로드 및 전처리
    try:
        dataset = ThreatTrajectoryDataset(
            data_dir=data_dir,
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
    data_to_save = {
        'obs_traj': dataset.obs_traj,
        'pred_traj': dataset.pred_traj,
        'obs_traj_rel': dataset.obs_traj_rel,
        'pred_traj_rel': dataset.pred_traj_rel,
        'loss_mask': dataset.loss_mask,
        'non_linear_ped': dataset.non_linear_ped,
        'seq_start_end': dataset.seq_start_end,
        'V_obs': dataset.V_obs,
        'A_obs': dataset.A_obs,
        'V_pred': dataset.V_pred,
        'A_pred': dataset.A_pred,
        'ped_masks': dataset.ped_masks,  # 보행자 마스크 추가
    }
    
    # 메모리 사용량 추정
    total_size = 0
    for key, value in data_to_save.items():
        if isinstance(value, torch.Tensor):
            size = value.element_size() * value.nelement() / (1024**2)  # MB
            total_size += size
            log(f"  → {key}: {value.shape} ({size:.2f} MB)")
        elif isinstance(value, list):
            list_size = sum(v.element_size() * v.nelement() for v in value if isinstance(v, torch.Tensor)) / (1024**2)
            total_size += list_size
            log(f"  → {key}: list with {len(value)} items ({list_size:.2f} MB)")
    
    log(f"  → 예상 총 크기: {total_size:.2f} MB")
    
    # .pt 파일로 저장
    log(f"\n.pt 파일 저장 중: {output_path}")
    save_start_time = time.time()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data_to_save, output_path)
    
    save_elapsed = time.time() - save_start_time
    actual_size = os.path.getsize(output_path) / (1024**2)
    
    log(f"저장 완료! (소요 시간: {save_elapsed:.2f}초)", "SUCCESS")
    log(f"  → 실제 파일 크기: {actual_size:.2f} MB")
    if total_size > 0:
        log(f"  → 압축률: {actual_size/total_size*100:.1f}%")
    
    return True

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
        output_path = os.path.join(data_dir, f'sdd_coupa_{split}_threat.pt')
        
        # 디렉토리 존재 확인
        if not os.path.exists(data_dir):
            log(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}", "ERROR")
            continue
        
        log(f"데이터 디렉토리: {data_dir}")
        log(f"출력 .pt 파일: {output_path}")
        
        # 전처리 및 저장
        try:
            success = preprocess_and_save(data_dir, output_path, min_ped=0)  # min_ped=0으로 변경하여 더 많은 시퀀스 추출
            if not success:
                log(f"전처리 실패: {split}", "ERROR")
                continue
        except Exception as e:
            log(f"전처리 중 오류 발생: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
        
        split_elapsed = time.time() - split_start_time
        log(f"\n{split.upper()} Split 완료! (총 소요 시간: {split_elapsed:.2f}초)", "SUCCESS")
    
    overall_elapsed = time.time() - overall_start_time
    
    log_section("전처리 완료 요약")
    log(f"총 소요 시간: {overall_elapsed:.2f}초 ({overall_elapsed/60:.2f}분)")
    log(f"\n생성된 파일:")
    
    total_size = 0
    for split in splits:
        output_path = os.path.join(base_dir, split, f'sdd_coupa_{split}_threat.pt')
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024**2)
            total_size += size_mb
            log(f"  ✓ {output_path}")
            log(f"    크기: {size_mb:.2f} MB")
        else:
            log(f"  ✗ {output_path} (생성되지 않음)", "WARNING")
    
    log(f"\n총 저장된 데이터 크기: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    log_section("모든 작업 완료!")

if __name__ == "__main__":
    main()

