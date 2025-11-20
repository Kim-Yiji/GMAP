#!/bin/bash
# Bookstore video0 threat score visualization
# Usage: ./visualize_bookstore_video0.sh <track_id> [sequence_idx]

TRACK_ID=${1:-20}  # Default track ID: 20
SEQUENCE_IDX=${2:-0}  # Default sequence index: 0

# Paths
DATA_DIR="/raid/guest/SDD_2beon/sdd_bookstore/test"
VIDEO_PATH="/raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets/video/bookstore/video0/video.mp4"
OUTPUT_PATH="./threat_viz_bookstore_video0_track${TRACK_ID}_seq${SEQUENCE_IDX}.mp4"

echo "Visualizing threat scores for track ID: ${TRACK_ID}"
echo "Data directory: ${DATA_DIR}"
echo "Video path: ${VIDEO_PATH}"
echo "Output: ${OUTPUT_PATH}"

cd /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN

python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir "${DATA_DIR}" \
    --target_track_id ${TRACK_ID} \
    --sequence_idx ${SEQUENCE_IDX} \
    --video_path "${VIDEO_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --scale 1000.0 \
    --labels_dir /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore \
    --class_size_csv /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore/class_sizes.csv \
    --visualize_mode all \
    --fps 10.0 \
    --device cuda:0

echo "Visualization complete: ${OUTPUT_PATH}"

