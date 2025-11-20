# Threat Score Calculation Pipeline

## Overview

This document describes the threat score calculation pipeline implemented in `utils/threat_score.py`. The pipeline computes continuous threat scores T_{ij} for each edge (i, j) in the graph, where i is a pedestrian and j is an obstacle.

## Pipeline Steps

### Step 1: Min-Max Normalization

Normalize each raw variable to the [0, 1] range using dataset-level minimum and maximum values:

```
x' = (x - x_min) / (x_max - x_min)
```

**Direction Reversal**: For variables where smaller values mean greater threat (distance d_{ij} and TTC_{ij}), reverse the direction after normalization:

```
d'_ij = 1 - (d_ij - d_min) / (d_max - d_min)
TTC'_ij = 1 - (TTC_ij - TTC_min) / (TTC_max - TTC_min)
```

This ensures all normalized values increase as threat increases.

**Features:**
- `d'_ij`: Normalized distance (reversed)
- `v'^+_ij`: Normalized approach velocity
- `size'_j`: Normalized obstacle size
- `TTC'_ij`: Normalized time-to-collision (reversed)

### Step 2: Linear Weighted Combination

Combine the normalized features using linear weighting:

```
u_{ij} = w_d * d'_ij + w_v * v'^+_ij + w_s * size'_j + w_T * TTC'_ij
```

**Default weights**: Distance-weighted [0.5, 0.25, 0.15, 0.1]
- Distance (w_d = 0.5): Has the largest impact based on empirical observations
- Approach velocity (w_v = 0.25): Second most important factor
- Obstacle size (w_s = 0.15): Moderate impact
- TTC (w_T = 0.1): Smallest impact

**Custom weights**: Can be provided as a list or tensor of shape (4,)

### Step 3: Sigmoid Transformation

Transform u_{ij} into a continuous probabilistic Threat Score:

```
Threat_{ij} = 1 / (1 + exp(-(u_{ij} - β) / τ))
```

**Parameters:**
- `τ` (tau): Temperature parameter controlling slope (default: 0.15)
  - Smaller values (0.1-0.2) create steeper transitions
  - Larger values create smoother transitions
- `β` (beta): Midpoint parameter controlling center (default: 0.5)
  - Controls where the sigmoid curve is centered

This ensures output values lie in [0, 1] and vary smoothly with input features.

### Step 4: Output

The resulting tensor has shape `(num_peds, num_peds, seq_len)` with values in [0, 1].

## Usage

```python
from utils.threat_score import compute_threat_score_batch

# Compute threat scores
threat_score, z_ij = compute_threat_score_batch(
    obs_traj,           # (num_peds, 2, seq_len) - absolute positions
    obs_traj_rel,       # (num_peds, 2, seq_len) - relative velocities
    obstacle_sizes=None,  # (num_peds,) or None - obstacle size values
    weights=None,       # [w_d, w_v, w_size, w_ttc] or None for distance-weighted [0.5, 0.25, 0.15, 0.1]
    tau=0.15,          # Temperature parameter
    beta=0.5           # Midpoint parameter
)

# threat_score: (num_peds, num_peds, seq_len) - threat scores in [0, 1]
# z_ij: (num_peds, num_peds, 4, seq_len) - raw features [d, v+, size, TTC]
```

## Feature Components (z_{ij})

Each edge (i, j) has a 4-dimensional feature vector:

1. **d_{ij}**: Distance between pedestrian i and obstacle j
   - Unit: Frame-based distance
   - Smaller values → higher threat

2. **v^{+}_{ij}**: Approach velocity
   - Unit: Frame-based velocity
   - Definition: max(0, -r_{ij} · v_{ij} / ||r_{ij}||)
   - Larger values → higher threat

3. **size_j**: Obstacle size
   - Values: 0.0 (human), 0.2 (similar to human), 0.7 (larger), 1.0 (much larger)
   - Larger values → higher threat

4. **TTC_{ij}**: Time-to-Collision
   - Unit: Frames
   - Definition: d_{ij} / max(ε, v^{+}_{ij})
   - Capped at T_max (default: 20 frames)
   - Smaller values → higher threat

## Obstacle Size Mapping

- **Human size (0.0)**: Pedestrian, Skater, Person pushing stroller
- **Similar to human (0.2)**: Biker, Bicycle, Cart, Shopping cart
- **Larger than human (0.7)**: Car, Vehicle
- **Much larger (1.0)**: Bus, Truck, Train

## Notes

- All calculations are performed in **frame units** (not seconds)
- Self-connections (i == j) are set to zero
- The pipeline does not modify the graph structure
- Numerical stability is ensured with epsilon values and clamping

## Example Output

```
Threat Score Results:
  Shape: torch.Size([236, 236, 11])
  Range: [0.0000, 0.8941]
  Mean: 0.1178
  Std: 0.0632
```

