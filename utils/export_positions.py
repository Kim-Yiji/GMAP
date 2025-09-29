"""
Export pedestrian positions and agent IDs per frame to CSV for overlay visualization.
Usage:
	python utils/export_positions.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from datasets.dataloader import TrajectoryDataset

def main():
	# 데이터셋 경로와 출력 파일명 지정
	data_dir = 'datasets/hotel/train'  # 필요에 따라 변경
	output_csv = 'group_allocation/hotel_framewise_positions.csv'

	print(f"Loading dataset from {data_dir} ...")
	dataset = TrajectoryDataset(data_dir=data_dir)
	print(f"Exporting framewise positions to {output_csv} ...")
	dataset.export_framewise_positions(output_csv)
	print("✅ Export complete.")

if __name__ == '__main__':
	main()
