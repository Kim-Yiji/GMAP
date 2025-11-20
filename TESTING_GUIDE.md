# Testing Guide for New Threat Score Algorithm

This guide explains how to test the updated threat score computation that uses the DMRGCN algorithm.

## Quick Test

Run the comprehensive test script:

```bash
python3 test_new_threat_score.py
```

This will test:
- Basic computation functions (distance, velocity, approach velocity, TTC)
- Threat feature computation
- Feature normalization
- Threat score computation
- Full pipeline with real data

## Step-by-Step Testing

### 1. Test Basic Functions

Test individual computation functions:

```python
from threat_score_viz.threat_score_computer import (
    compute_distance,
    compute_approach_velocity,
    compute_ttc,
    get_obstacle_size
)

# Test distance
distance = compute_distance((0, 0), (3, 4))
print(f"Distance: {distance}")  # Should be 5.0

# Test obstacle size
size = get_obstacle_size('car')
print(f"Car size: {size}")  # Should be 0.7
```

### 2. Test Threat Feature Computation

Test the 4 threat features:

```python
from threat_score_viz.threat_score_computer import compute_threat_features

target_pos = (10.0, 10.0)
target_prev_pos = (9.0, 10.0)
obstacle_pos = (12.0, 10.0)
obstacle_prev_pos = (13.0, 10.0)

d_ij, v_plus_ij, size_j, ttc_ij = compute_threat_features(
    target_pos, target_prev_pos,
    obstacle_pos, obstacle_prev_pos,
    obstacle_size=0.0
)

print(f"Distance: {d_ij:.2f}")
print(f"Approach velocity: {v_plus_ij:.3f}")
print(f"Obstacle size: {size_j:.1f}")
print(f"TTC: {ttc_ij:.2f}")
```

### 3. Test with Real Data (Single Frame)

Test threat score computation for a single frame:

```python
from threat_score_viz.threat_score_computer import (
    compute_threat_scores_for_frame,
    build_object_position_history
)
from threat_score_viz.data_parser import get_object_positions_per_frame
from threat_score_viz.target_selector import get_target_positions

annotation_path = 'sdd_bookstore/test/bookstore_video0_test.txt'
target_id = 176  # Or use auto-select
frame_id = 2500

# Load data
frame_data = get_object_positions_per_frame(annotation_path)
target_positions = get_target_positions(annotation_path, target_id)
object_positions = build_object_position_history(annotation_path)

# Compute threat scores
threat_scores = compute_threat_scores_for_frame(
    frame_id, target_id, frame_data,
    target_positions, object_positions
)

# Display results
for obj_id, score, pos, features in sorted(threat_scores, key=lambda x: x[1], reverse=True)[:5]:
    d, v, s, ttc = features
    print(f"Object {obj_id}: score={score:.3f}, d={d:.2f}, v+={v:.3f}, size={s:.1f}, TTC={ttc:.2f}")
```

### 4. Test Full Video Processing

Test the complete visualization pipeline:

```bash
# List available targets
python3 -m threat_score_viz.main \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --list-candidates

# Process a small sample (10 frames)
python3 -m threat_score_viz.main \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/test_sample.mp4 \
  --output-metadata output/test_sample.json \
  --start-frame 2500 \
  --end-frame 2510
```

### 5. Test with Custom Parameters

Test with different algorithm parameters:

```bash
# Use custom weights
python3 -m threat_score_viz.main \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/test_custom.mp4 \
  --output-metadata output/test_custom.json \
  --weights 0.6 0.2 0.15 0.05 \
  --tau 0.1 \
  --beta 0.5 \
  --start-frame 2500 \
  --end-frame 2510
```

## Expected Results

### Feature Values

- **Distance (d_ij)**: Positive values, typically 0-100 in annotation coordinates
- **Approach velocity (v+_ij)**: Non-negative, typically 0-5 pixels/frame
- **Obstacle size (size_j)**: 0.0 (human), 0.2 (bike), 0.7 (car), 1.0 (bus)
- **TTC (ttc_ij)**: Positive values, capped at 20.0 frames

### Threat Scores

- **Range**: [0, 1]
- **High threat**: > 0.7 (close, approaching, large obstacle, low TTC)
- **Medium threat**: 0.4 - 0.7
- **Low threat**: < 0.4 (far, not approaching, small obstacle, high TTC)

### Default Parameters

- **Weights**: [0.5, 0.25, 0.15, 0.1] (distance-weighted)
- **Tau**: 0.15 (sigmoid temperature)
- **Beta**: 0.5 (sigmoid midpoint)
- **TTC max**: 20.0 frames

## Comparison with Old Algorithm

The new algorithm differs from the old one in several ways:

1. **Features**: 
   - Old: distance, velocity_diff, heading_alignment, class_interaction
   - New: distance, approach_velocity, obstacle_size, TTC

2. **Normalization**:
   - Old: Simple normalization with fixed max values
   - New: Min-max normalization with direction reversal for distance and TTC

3. **Weights**:
   - Old: [1.0, 1.0, 1.0, 1.0] (uniform)
   - New: [0.5, 0.25, 0.15, 0.1] (distance-weighted)

4. **Sigmoid**:
   - Old: `sigmoid(weighted_sum)`
   - New: `sigmoid((weighted_sum - beta) / tau)`

## Troubleshooting

### Issue: Threat scores are all very low (< 0.1)

**Possible causes**:
- Objects are very far apart
- No approach velocity (objects moving away or parallel)
- Normalization range is too large

**Solutions**:
- Check if objects are actually close in the frame
- Verify velocity computation (need previous positions)
- Adjust tau parameter (lower = steeper sigmoid)

### Issue: Threat scores are all very high (> 0.9)

**Possible causes**:
- Objects are very close
- High approach velocities
- Normalization range is too small

**Solutions**:
- Verify distance computation
- Check approach velocity values
- Adjust tau parameter (higher = smoother sigmoid)

### Issue: Scores don't change between frames

**Possible causes**:
- Missing previous positions (can't compute velocity)
- Objects not moving
- Normalization computed per-frame (should be consistent)

**Solutions**:
- Ensure objects have previous frame positions
- Check if objects are actually moving
- Verify normalization is working correctly

## Visual Inspection

After processing, check the output video:

1. **High threat** (red): Objects close to target, approaching
2. **Medium threat** (orange): Objects at medium distance, moderate approach
3. **Low threat** (green): Objects far away, not approaching

The target should be marked with a cyan/magenta circle.

## Next Steps

After testing:

1. Process a longer video segment to see threat scores over time
2. Compare results with the old algorithm (if you have old outputs)
3. Adjust parameters (weights, tau, beta) based on your use case
4. Use obstacle size information if you have object type labels

## Example Output

```
Testing New DMRGCN-Based Threat Score Algorithm
============================================================

Testing Basic Functions
============================================================
1. Testing distance computation...
   ✓ Distance between (0.0, 0.0) and (3.0, 4.0): 5.00 (expected: 5.0)
...

Testing Full Pipeline with Real Data
============================================================
1. Selecting target...
   ✓ Selected target: 176 (appears in 1234 frames)
2. Loading annotation data...
   ✓ Loaded 5000 frames
   ✓ Target appears in 1234 frames
3. Finding test frame...
   ✓ Testing with frame 2500
4. Computing threat scores...
   ✓ Computed threat scores for 15 obstacles

   Top 5 threat scores in frame 2500:
   1. Object 177: score=0.823, pos=(36.6, 36.7)
      Features: d=1.34, v+=0.245, size=0.0, TTC=5.47
   ...

✓ ALL TESTS PASSED!
```

