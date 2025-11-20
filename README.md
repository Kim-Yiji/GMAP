# Threat Score Visualization for Stanford Drone Dataset (SDD)

This project computes and visualizes threat scores for pedestrians in the Stanford Drone Dataset. It uses heuristic calculations to determine how "threatening" or "dangerous" obstacles (other pedestrians, vehicles, bicycles, etc.) are relative to a target pedestrian.

## Overview

The system processes video frames and annotations from the SDD dataset to:
1. **Calculate threat scores** for each obstacle relative to a target pedestrian
2. **Visualize relationships** as a graph where nodes represent objects and edges represent threat scores
3. **Generate annotated videos** with overlaid threat scores and graph visualizations
4. **Export metadata** containing frame-by-frame threat score information

## Threat Score Calculation

### Core Formula

The threat score is computed using a weighted sum of relational features, passed through a sigmoid activation function:

```
threat_score = sigmoid(w1*f1 + w2*f2 + w3*f3 + w4*f4)
```

Where:
- `w1, w2, w3, w4` are weights (default: all 1.0)
- `f1, f2, f3, f4` are normalized relational features
- `sigmoid(x) = 1 / (1 + e^(-x))` ensures output is in [0, 1] range

### Relational Features

#### f1: Distance Feature
- **Purpose**: Measures spatial proximity between target and obstacle
- **Calculation**: 
  ```
  distance = sqrt((target_x - obstacle_x)² + (target_y - obstacle_y)²)
  f1 = normalize_distance(distance)
  ```
- **Normalization**: 
  ```
  normalize_distance(d) = 1 - min(d / max_distance, 1.0)
  ```
  - Closer objects → higher normalized value → higher threat
  - `max_distance` default: 100.0 meters

#### f2: Velocity Difference Feature
- **Purpose**: Measures relative speed between target and obstacle
- **Calculation**:
  ```
  target_velocity = (target_pos[t] - target_pos[t-1]) / dt
  obstacle_velocity = (obstacle_pos[t] - obstacle_pos[t-1]) / dt
  velocity_diff = ||target_velocity - obstacle_velocity||
  f2 = normalize_velocity_difference(velocity_diff)
  ```
- **Normalization**:
  ```
  normalize_velocity_diff(v) = min(v / max_vel_diff, 1.0)
  ```
  - Larger velocity differences → higher normalized value → higher threat
  - `max_vel_diff` default: 10.0 m/s

#### f3: Heading Alignment Feature
- **Purpose**: Measures whether target and obstacle are moving toward each other
- **Calculation**:
  ```
  target_heading = atan2(target_vy, target_vx)
  obstacle_heading = atan2(obstacle_vy, obstacle_vx)
  relative_heading = obstacle_heading - target_heading
  alignment = cos(relative_heading)  # -1 (opposite) to +1 (same direction)
  f3 = (1 - alignment) / 2  # Normalize to [0, 1]
  ```
- **Interpretation**:
  - Objects moving toward each other (head-on) → alignment ≈ -1 → f3 ≈ 1 (high threat)
  - Objects moving in same direction → alignment ≈ +1 → f3 ≈ 0 (low threat)

#### f4: Object Class Interaction Feature
- **Purpose**: Models different threat levels for different object types
- **Current Implementation**: Placeholder using distance-based heuristic
- **Calculation**:
  ```
  f4 = normalize_distance(distance)  # Same as f1 for now
  ```
- **Future Enhancement**: Could incorporate object class (pedestrian, car, bicycle, etc.)

### Example Calculation

For a target at (24.5, 28.6) and obstacle at (26.2, 30.1):
1. **Distance**: `sqrt((26.2-24.5)² + (30.1-28.6)²) = 2.34 meters`
   - `f1 = 1 - min(2.34/100, 1) = 0.977`

2. **Velocity Difference**: Target moving at (0.5, 0.3) m/s, obstacle at (0.8, -0.2) m/s
   - `velocity_diff = sqrt((0.8-0.5)² + (-0.2-0.3)²) = 0.58 m/s`
   - `f2 = min(0.58/10, 1) = 0.058`

3. **Heading Alignment**: Target heading 30°, obstacle heading 210° (opposite)
   - `alignment = cos(210° - 30°) = cos(180°) = -1`
   - `f3 = (1 - (-1)) / 2 = 1.0`

4. **Object Class**: Using distance-based heuristic
   - `f4 = f1 = 0.977`

5. **Threat Score**: With all weights = 1.0
   - `sum = 1.0*0.977 + 1.0*0.058 + 1.0*1.0 + 1.0*0.977 = 3.012`
   - `threat_score = sigmoid(3.012) = 1 / (1 + e^(-3.012)) = 0.953` (high threat!)

## Coordinate Transformation

### Problem

SDD annotations are in **world coordinates** (meters), while videos are in **pixel coordinates**. We need to transform annotation coordinates to pixel coordinates for visualization.

### Homography-Based Transformation (Preferred)

When a homography file (`H_SDD.txt`) is available, we use calibration data to compute accurate transformations.

#### Homography File Format

The `H_SDD.txt` file contains calibration data for each scene:
- **Pixel coordinates**: Known pixel positions in the video
- **Real-world measurements**: Corresponding distances in meters
- **Ratio**: Pixels per meter conversion factor

Example entry for `bookstore_0`:
```
bookstore_0.jpg  Bookstore  0  A  44  0  13.4112  349  15  349.32  0.03839
```
- 13.4112 meters = 349.32 pixels
- Pixels per meter = 349.32 / 13.4112 ≈ 26.09 pixels/meter

#### Transformation Calculation

1. **Extract calibration data** from homography file for the scene
2. **Calculate pixels per meter**: `pixels_per_meter = pixel_distance / meters`
3. **Scale coordinates**: 
   ```
   pixel_x = annotation_x * pixels_per_meter
   pixel_y = annotation_y * pixels_per_meter
   ```
4. **Apply offset**: Map annotation origin/minimum to video margins
   ```
   offset_x = margin - (annotation_min_x * pixels_per_meter)
   offset_y = margin - (annotation_min_y * pixels_per_meter)
   final_x = pixel_x + offset_x
   final_y = pixel_y + offset_y
   ```

### Fallback Transformation

If no homography file is found, we use a standard transformation:
1. **Calculate scale**: Map annotation range to video dimensions (with margins)
2. **Preserve position**: Keep objects in their relative upper-left region
3. **Apply manual offset adjustments**: Fine-tune alignment if needed

## Visualization

### Graph Representation

The visualization represents the scene as a graph:
- **Nodes**: Objects (target + obstacles)
- **Edges**: Threat scores between target and obstacles

### Visual Elements

#### Target Marker
- **Shape**: Concentric circles (outer black, cyan ring, magenta fill, white center)
- **Size**: Radius 10 pixels (reduced to not obscure the person)
- **Label**: "TARGET ID: {target_id}" above the marker
- **Color**: Bright cyan/magenta for high visibility

#### Obstacle Nodes
- **Shape**: Circles with radius 4-8 pixels (based on threat score)
- **Color**: Gradient based on threat score
  - **Green** (low threat, < 0.4)
  - **Yellow/Orange** (medium threat, 0.4-0.7)
  - **Red** (high threat, ≥ 0.7)
- **Label**: "id : {obstacle_id}" above the node
- **Transparency**: Alpha = 0.5 (semi-transparent)

#### Edges (Threat Score Lines)
- **Thickness**: 1-3 pixels (based on threat score)
- **Color**: Same gradient as nodes (green → yellow → red)
- **Transparency**: Alpha = 0.4 (more transparent than nodes)
- **Label**: "threat_score : {score}" for longer edges (distance > 50px)

### Drawing Pipeline

1. **Load frame** from video
2. **Transform coordinates** from annotation space to pixel space
3. **Draw target marker** at target position
4. **For each obstacle**:
   - Calculate threat score
   - Draw edge from target to obstacle
   - Draw obstacle node
   - Draw labels (ID and threat score)
5. **Save frame** to output video

## File Structure

```
threat_score/
├── README.md                          # This file
├── H_SDD.txt                          # Homography calibration data
├── threat_score_viz/                  # Main package
│   ├── __init__.py                    # Package initialization
│   ├── main.py                        # CLI entry point
│   ├── data_parser.py                 # Annotation file parsing
│   ├── video_utils.py                 # Video loading utilities
│   ├── target_selector.py             # Target pedestrian selection
│   ├── threat_score_computer.py       # Threat score calculation
│   ├── homography_parser.py           # Homography file parsing
│   └── visualizer.py                  # Visualization and video processing
├── sdd_bookstore/                     # SDD dataset (bookstore scene)
│   ├── bookstore_vid/                 # Video files
│   └── test/                          # Test annotations
└── output/                            # Output videos and metadata
```

## Usage

### Basic Usage

```bash
python3 -m threat_score_viz \
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \
    --output-video output/video.mp4 \
    --output-metadata output/metadata.json
```

### With Target Selection

```bash
python3 -m threat_score_viz \
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \
    --output-video output/video.mp4 \
    --output-metadata output/metadata.json \
    --target-id 212
```

### With Custom Weights

```bash
python3 -m threat_score_viz \
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \
    --output-video output/video.mp4 \
    --output-metadata output/metadata.json \
    --weights 2.0 1.0 1.5 0.5  # w1, w2, w3, w4
```

### Frame Range

```bash
python3 -m threat_score_viz \
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \
    --output-video output/video.mp4 \
    --output-metadata output/metadata.json \
    --start-frame 5000 \
    --end-frame 5150  # 5 seconds at 30 fps
```

### Coordinate Adjustment

```bash
python3 -m threat_score_viz \
    --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
    --annotations sdd_bookstore/test/bookstore_video0_test.txt \
    --output-video output/video.mp4 \
    --output-metadata output/metadata.json \
    --offset-x -90 \
    --offset-y 100
```

## Metadata Format

The output metadata file (JSON) contains frame-by-frame information:

```json
{
  "video_path": "sdd_bookstore/bookstore_vid/video0/video.mp4",
  "annotation_path": "sdd_bookstore/test/bookstore_video0_test.txt",
  "target_id": 212,
  "frames": [
    {
      "frame_id": 5000,
      "target_id": 212,
      "target_position": [24.49, 28.57],
      "interactions": [
        {
          "object_id": 20,
          "score": 0.753,
          "position": [29.06, 35.26]
        },
        {
          "object_id": 28,
          "score": 0.642,
          "position": [34.46, 7.03]
        }
      ]
    }
  ]
}
```

## Key Algorithms

### Target Selection

The system can automatically select a target pedestrian using:
1. **Frame presence**: Prefer pedestrians present in many frames
2. **Central location**: Prefer pedestrians near the center of activity
3. **Combined score**: Weighted combination of presence and centrality

### Velocity Calculation

Velocities are computed using finite differences:
```python
velocity_x = (position_x[t] - position_x[t-1]) / dt
velocity_y = (position_y[t] - position_y[t-1]) / dt
```

Where `dt = 1/fps` (typically 1/30 seconds for 30 fps video).

### History Building

To compute velocities, we maintain a history of positions:
- Store last N positions for each object (default: 5 frames)
- Use linear interpolation for missing frames
- Handle objects entering/leaving the scene

## Parameters

### Threat Score Weights
- **w1** (distance): Default 1.0
- **w2** (velocity difference): Default 1.0
- **w3** (heading alignment): Default 1.0
- **w4** (object class): Default 1.0

### Normalization Parameters
- **max_distance**: Maximum distance for normalization (default: 100.0 meters)
- **max_vel_diff**: Maximum velocity difference for normalization (default: 10.0 m/s)

### Visualization Parameters
- **Target marker radius**: 10 pixels
- **Obstacle node radius**: 4-8 pixels (based on threat score)
- **Edge thickness**: 1-3 pixels (based on threat score)
- **Alpha (transparency)**: 0.4 for edges, 0.5 for nodes

## Limitations and Future Work

### Current Limitations
1. **Object class feature (f4)**: Currently uses distance-based heuristic
2. **Fixed weights**: All features weighted equally by default
3. **No temporal smoothing**: Threat scores computed independently per frame
4. **Manual offset adjustment**: May be needed for perfect alignment

### Future Enhancements
1. **Object class integration**: Use actual object types (pedestrian, car, bicycle)
2. **Adaptive weights**: Learn optimal weights from data
3. **Temporal smoothing**: Average threat scores over time windows
4. **Automatic calibration**: Improve coordinate transformation accuracy
5. **Interactive visualization**: Real-time adjustment of parameters
6. **Multi-target support**: Visualize threat scores for multiple targets simultaneously

## References

- **Stanford Drone Dataset**: https://cvgl.stanford.edu/projects/uav_data/
- **Homography calibration**: Uses calibration data from SDD for coordinate transformation
- **Sigmoid function**: Standard logistic function for normalization

## License

[Add your license information here]

## Authors

[Add author information here]
