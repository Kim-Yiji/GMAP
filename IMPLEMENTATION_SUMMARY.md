# Threat Score Visualization - Implementation Summary

## Overview

This document summarizes the step-by-step implementation of the threat score visualization system for the SDD (Stanford Drone Dataset).

## Step 1: Data Parser Module ✅

**File**: `threat_score_viz/data_parser.py`

**Functionality**:
- Load SDD annotation files (format: `frame_id \t ped_id \t x \t y`)
- Organize annotations by frame
- Extract object positions per frame
- Get available object IDs and frame ranges

**Key Functions**:
- `load_annotations()`: Load annotation file into numpy array
- `organize_by_frame()`: Organize annotations by frame_id
- `get_object_positions_per_frame()`: Convenience function to load and organize
- `get_available_object_ids()`: Get list of unique object IDs
- `get_frame_range()`: Get min/max frame IDs

**Test Results**:
- ✓ Loaded 249,850 annotation entries
- ✓ Organized into 13,335 frames
- ✓ Identified 237 unique object IDs
- ✓ Frame range: 0 to 13,334

## Step 2: Video Alignment Verification ✅

**File**: `threat_score_viz/video_utils.py`

**Functionality**:
- Load video properties (frame count, FPS, resolution)
- Verify video-annotation alignment
- Load individual video frames

**Key Functions**:
- `get_video_properties()`: Get video metadata
- `verify_annotation_video_alignment()`: Verify frame count matches
- `load_video_frame()`: Load specific frame from video

**Verification Results**:
- ✓ Video frames: 13,335
- ✓ Annotation frames: 13,335 (range: 0-13,334)
- ✓ Perfect alignment confirmed
- ✓ Frame IDs match video frames directly

## Step 3: Target Pedestrian Selection ✅

**File**: `threat_score_viz/target_selector.py`

**Functionality**:
- Get object presence statistics
- Find best target candidates (objects with most frames)
- Validate target IDs
- Get target positions across frames
- Auto-select or manually select target

**Key Functions**:
- `get_object_presence_stats()`: Statistics for each object
- `find_best_target_candidates()`: Find objects with most frames
- `validate_target()`: Validate target exists
- `get_target_positions()`: Get target positions per frame
- `select_target()`: Select target (auto or manual)

**Test Results**:
- ✓ Found 237 unique objects
- ✓ Best candidate: Object 176 (10,835 frames, range: 2500-13334)
- ✓ Target validation working
- ✓ Position tracking working

## Step 4: Threat Score Computation ✅

**File**: `threat_score_viz/threat_score_computer.py`

**Functionality**:
- Compute 4 relational features:
  - **f1**: Distance (normalized, closer = higher)
  - **f2**: Velocity difference (normalized, higher diff = higher)
  - **f3**: Heading alignment (same direction = higher)
  - **f4**: Object class interaction (placeholder, distance-based)
- Compute threat score using weighted sum + sigmoid
- Process frames to compute threat scores for all obstacles

**Key Functions**:
- `compute_distance()`: Euclidean distance
- `compute_velocity_magnitude()`: Speed from position change
- `compute_heading()`: Direction angle
- `compute_heading_alignment()`: Direction similarity
- `compute_relational_features()`: Compute all 4 features
- `compute_threat_score()`: Weighted sum + sigmoid
- `compute_threat_scores_for_frame()`: Compute scores for all obstacles in a frame

**Threat Score Formula**:
```
threat_score = sigmoid(w1*f1 + w2*f2 + w3*f3 + w4*f4)
```
Where w1 = w2 = w3 = w4 = 1.0 (uniform weights)

**Test Results**:
- ✓ Distance computation: Working
- ✓ Velocity computation: Working
- ✓ Heading computation: Working (45° angle computed correctly)
- ✓ Heading alignment: Working (same=1.0, opposite=0.0)
- ✓ Feature normalization: Working
- ✓ Threat scores: Computed for 17 obstacles in frame 2500
- ✓ Top threat: Object 177 with score 0.879

## Step 5: Visualization ✅

**File**: `threat_score_viz/visualizer.py`

**Functionality**:
- Draw threat score annotations on video frames
- Color-code threat scores (red=high, orange=medium, green=low)
- Draw target marker
- Process entire videos with annotations
- Save annotated videos and metadata

**Key Functions**:
- `draw_threat_score_on_frame()`: Draw score annotation for one obstacle
- `draw_target_marker()`: Draw target marker
- `process_video_frame()`: Process single frame with all annotations
- `create_frame_metadata()`: Create metadata dictionary for a frame
- `process_video_with_threat_scores()`: Process entire video

**Visualization Features**:
- Color-coded text:
  - Red: High threat (score >= 0.7)
  - Orange: Medium threat (0.4 <= score < 0.7)
  - Green: Low threat (score < 0.4)
- Target marker: Cyan circle with "TARGET" label
- Text format: `ID:XXX 0.XX` (object ID and threat score)

**Test Results**:
- ✓ Single frame visualization: Working
- ✓ Video processing: Working (processed 11 frames in sample)
- ✓ Metadata generation: Working (JSON format with all required fields)
- ✓ Output files created: Video (84 KB) and metadata (77 KB)

## Step 6: CLI Interface ✅

**File**: `threat_score_viz/main.py`

**Functionality**:
- Command-line interface for processing videos
- List target candidates
- Process videos with various options
- Verify alignment
- Customize weights and parameters

**Key Features**:
- Auto-select or manually select target
- Process entire video or specific frame range
- Customize threat score weights
- Verify video-annotation alignment
- Save both video and metadata outputs

**Usage Examples**:
```bash
# List candidates
python -m threat_score_viz --annotations <path> --list-candidates

# Process video
python -m threat_score_viz \
  --video <video_path> \
  --annotations <annotation_path> \
  --output-video <output_path> \
  --output-metadata <metadata_path>
```

## Output Format

### Annotated Video
- Threat score annotations for each obstacle
- Color-coded by threat level
- Target marker
- Preserves original video quality and FPS

### Metadata JSON
```json
{
  "video_path": "...",
  "annotation_path": "...",
  "target_id": 176,
  "video_properties": {...},
  "processing_settings": {...},
  "frames": [
    {
      "frame_id": 0,
      "target_id": 176,
      "target_position": [x, y],
      "interactions": [
        {
          "object_id": 177,
          "score": 0.879,
          "position": [x, y],
          "features": {
            "f1_distance": 0.986,
            "f2_velocity_diff": 0.0,
            "f3_heading_alignment": 0.5,
            "f4_class_interaction": 0.493
          }
        }
      ]
    }
  ]
}
```

## Module Structure

```
threat_score_viz/
├── __init__.py          # Package initialization
├── __main__.py          # Module entry point
├── data_parser.py       # Step 1: Annotation parsing
├── video_utils.py       # Step 2: Video utilities
├── target_selector.py   # Step 3: Target selection
├── threat_score_computer.py  # Step 4: Threat score computation
├── visualizer.py        # Step 5: Visualization
└── main.py              # Step 6: CLI interface
```

## Testing

All steps have been tested individually:
- ✓ `test_step1_parser.py`: Data parser tests
- ✓ `test_video_alignment.py`: Video alignment verification
- ✓ `test_step2_target_selection.py`: Target selection tests
- ✓ `test_step3_threat_score.py`: Threat score computation tests
- ✓ `test_step4_visualization.py`: Visualization tests

## Next Steps

1. **Replace placeholder features**: When the extended DMRGCN model (4 relational graphs) is trained, replace the heuristic threat score computation with learned weights.

2. **Object class information**: Incorporate actual object class information into f4 feature (currently using distance-based placeholder).

3. **Performance optimization**: For long videos, consider:
   - Batch processing
   - Multi-threading
   - GPU acceleration

4. **Advanced visualization**: Add options for:
   - Trajectory lines
   - Heat maps
   - Interactive visualization

## Requirements

- Python 3.8+
- opencv-python
- numpy
- tqdm

## Usage

See `USAGE_EXAMPLE.md` for detailed usage examples and command-line options.

