# Threat Score Visualization - Usage Examples

## Overview

This package computes and visualizes threat scores for pedestrian trajectories in the SDD (Stanford Drone Dataset). Threat scores are computed based on 4 relational features and overlaid on video frames.

## Installation

Make sure you have the required dependencies:
```bash
pip install opencv-python numpy tqdm
```

## Quick Start

### 1. List Available Target Candidates

First, find good target pedestrians (those that appear in many frames):

```bash
python3 -m threat_score_viz.main \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --list-candidates
```

This will show a list of objects sorted by frame count, helping you choose a good target.

### 2. Process Video with Auto-Selected Target

Process the entire video with automatic target selection:

```bash
python3 -m threat_score_viz.main \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/annotated_video.mp4 \
  --output-metadata output/metadata.json
```

### 3. Process Specific Frame Range with Custom Target

Process a specific range of frames with a manually selected target:

```bash
python3 -m threat_score_viz.main \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/annotated_video_2500_3000.mp4 \
  --output-metadata output/metadata_2500_3000.json \
  --target-id 176 \
  --start-frame 2500 \
  --end-frame 3000
```

### 4. Customize Threat Score Weights

Adjust the weights for different features:

```bash
python3 -m threat_score_viz.main \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/annotated_video.mp4 \
  --output-metadata output/metadata.json \
  --weights 2.0 1.0 1.5 0.5
```

The weights correspond to:
- w1: Distance feature weight
- w2: Velocity difference feature weight
- w3: Heading alignment feature weight
- w4: Object class interaction feature weight

### 5. Verify Video-Annotation Alignment

Before processing, verify that the video and annotation files are aligned:

```bash
python3 -m threat_score_viz.main \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/annotated_video.mp4 \
  --output-metadata output/metadata.json \
  --verify-alignment
```

## Command Line Options

### Required Arguments
- `--annotations`: Path to annotation file (required)
- `--video`: Path to input video file (required for processing)
- At least one of `--output-video` or `--output-metadata` must be provided

### Optional Arguments
- `--target-id`: Target pedestrian ID (if not provided, auto-selects)
- `--list-candidates`: List available target candidates and exit
- `--start-frame`: Starting frame (default: 0)
- `--end-frame`: Ending frame (default: process all frames)
- `--weights`: Weights for threat score features (default: 1.0 1.0 1.0 1.0)
- `--max-distance`: Maximum distance for normalization (default: 100.0)
- `--max-vel-diff`: Maximum velocity difference for normalization (default: 10.0)
- `--scale`: Scale factor for text size (default: 1.0)
- `--no-target-marker`: Do not draw target marker
- `--verify-alignment`: Verify video-annotation alignment before processing

## Output Format

### Annotated Video

The output video contains:
- Threat score annotations for each obstacle (ID and score)
- Color-coded text:
  - Red: High threat (score >= 0.7)
  - Orange: Medium threat (0.4 <= score < 0.7)
  - Green: Low threat (score < 0.4)
- Target marker (cyan circle with "TARGET" label)

### Metadata JSON

The metadata file contains frame-by-frame information:

```json
{
  "video_path": "...",
  "annotation_path": "...",
  "target_id": 176,
  "video_properties": {
    "frame_count": 13335,
    "fps": 29.97,
    "width": 1416,
    "height": 1080
  },
  "processing_settings": {
    "weights": [1.0, 1.0, 1.0, 1.0],
    "max_distance": 100.0,
    "max_vel_diff": 10.0,
    "start_frame": 0,
    "end_frame": 13334
  },
  "frames": [
    {
      "frame_id": 0,
      "target_id": 176,
      "target_position": [35.24, 36.88],
      "interactions": [
        {
          "object_id": 177,
          "score": 0.879,
          "position": [36.6, 36.7],
          "features": {
            "f1_distance": 0.986,
            "f2_velocity_diff": 0.0,
            "f3_heading_alignment": 0.5,
            "f4_class_interaction": 0.493
          }
        },
        ...
      ]
    },
    ...
  ]
}
```

## Python API Usage

You can also use the package programmatically:

```python
from threat_score_viz.visualizer import process_video_with_threat_scores

# Process video
stats = process_video_with_threat_scores(
    video_path='sdd_bookstore/bookstore_vid/video0/video.mp4',
    annotation_path='sdd_bookstore/test/bookstore_video0_test.txt',
    output_video_path='output/annotated_video.mp4',
    output_metadata_path='output/metadata.json',
    target_id=176,
    start_frame=2500,
    end_frame=3000
)

print(f"Processed {stats['frames_processed']} frames")
```

## Threat Score Calculation

Threat scores are computed using:

```
threat_score = sigmoid(w1*f1 + w2*f2 + w3*f3 + w4*f4)
```

Where:
- **f1**: Normalized distance feature (closer = higher)
- **f2**: Normalized velocity difference (higher difference = higher)
- **f3**: Heading alignment (same direction = higher)
- **f4**: Object class interaction (placeholder, currently distance-based)

Default weights: w1 = w2 = w3 = w4 = 1.0

## Module Structure

- `data_parser.py`: Load and parse annotation files
- `target_selector.py`: Select and validate target pedestrians
- `threat_score_computer.py`: Compute threat scores using relational features
- `visualizer.py`: Overlay threat scores on video frames
- `video_utils.py`: Video loading and alignment utilities
- `main.py`: CLI interface

## Testing

Run the test scripts to verify functionality:

```bash
# Test data parser
python3 test_step1_parser.py

# Test target selection
python3 test_step2_target_selection.py

# Test threat score computation
python3 test_step3_threat_score.py

# Test visualization
python3 test_step4_visualization.py
```

## Notes

- The threat score computation uses heuristic/placeholder features. In the future, this will be replaced with learned weights from the extended DMRGCN model.
- Object class information is not currently used (f4 is a placeholder).
- Video processing can be memory-intensive for long videos. Consider processing in chunks using `--start-frame` and `--end-frame`.

