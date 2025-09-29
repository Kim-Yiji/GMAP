#!/usr/bin/env python3
"""
시각화용 더미 PKL 파일 생성 스크립트
"""

import os
import pickle
import numpy as np


def main():
    """메인 함수 - 더미 PKL 파일 생성"""
    print("🚀 시각화용 더미 PKL 파일 생성")
    
    output_dir = '../ETH-UCY-Trajectory-Visualizer/pred_traj_dump'
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['hotel', 'eth', 'univ', 'zara1', 'zara2']
    
    for dataset_name in datasets:
        print(f"\n🎯 {dataset_name} 데이터셋용 PKL 생성")
        
        # 더미 예측 데이터 생성
        predictions = []
        groups = []
        metadata = []
        
        np.random.seed(42)  # 재현 가능한 결과
        
        for seq_idx in range(30):  # 30개 시퀀스
            num_agents = np.random.randint(2, 5)  # 2-4 에이전트
            pred_len = 12  # 예측 길이
            
            seq_predictions = []
            seq_groups = []
            
            for agent_idx in range(num_agents):
                # 현실적인 궤적 생성 (곡선 움직임)
                t = np.linspace(0, 1, pred_len)
                
                # 기본 궤적 with 곡선
                x = t * 6.0 + np.sin(t * np.pi * 2) * 1.5 + agent_idx * 2.0
                y = t * 4.0 + np.cos(t * np.pi * 1.5) * 1.0 + agent_idx * 1.5
                
                trajectory = np.column_stack([x, y])  # [pred_len, 2]
                
                # 노이즈 추가
                noise = np.random.normal(0, 0.15, trajectory.shape)
                trajectory += noise
                
                seq_predictions.append(trajectory)
                
                # 그룹 할당 (2-3명씩 그룹)
                group_id = agent_idx // 2
                seq_groups.append(group_id)
            
            predictions.append(seq_predictions)
            groups.append(np.array(seq_groups))
            
            metadata.append({
                'seq_idx': seq_idx,
                'num_agents': num_agents,
                'pred_length': pred_len
            })
        
        # 시각화 데이터 구조 생성
        visualization_data = {
            'predictions': predictions,
            'groups': groups,
            'metadata': metadata,
            'dataset': dataset_name,
            'format_info': {
                'description': 'Dummy GMAP predictions with group info',
                'access_pattern': 'data["predictions"][seq_idx][agent_idx] -> [T_pred, 2]',
                'group_pattern': 'data["groups"][seq_idx][agent_idx] -> group_id'
            }
        }
        
        # PKL 파일 저장
        output_path = f'{output_dir}/GMAP_{dataset_name}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(visualization_data, f)
        
        total_agents = sum(len(seq) for seq in predictions)
        print(f"✅ 저장 완료: {output_path}")
        print(f"   시퀀스: {len(predictions)}, 총 에이전트: {total_agents}")
    
    print(f"\n🎉 모든 더미 PKL 파일 생성 완료!")
    print(f"📁 출력 폴더: {output_dir}")


if __name__ == '__main__':
    main()