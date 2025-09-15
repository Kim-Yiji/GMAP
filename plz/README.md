# Unified Trajectory Prediction Model (PLZ)

A unified trajectory prediction model that combines the strengths of DMRGCN and GPGraph, incorporating velocity, acceleration, relative displacement, and group dynamics for enhanced pedestrian trajectory prediction.

## Overview

This project merges two state-of-the-art trajectory prediction approaches:
- **DMRGCN**: Dynamic Multi-Relational Graph Convolutional Network for velocity and relative displacement modeling
- **GPGraph**: Group-aware Pedestrian Graph for modeling group interactions and dynamics

The unified model leverages:
- ✅ **Velocity and Acceleration**: Enhanced motion understanding through velocity and acceleration features
- ✅ **Relative Displacement**: Spatial relationship modeling between pedestrians
- ✅ **Group Dynamics**: Automatic group detection and intra/inter-group interaction modeling
- ✅ **Multi-scale Interactions**: Individual, group, and scene-level social interactions

## Key Features

### Enhanced DMRGCN
- Multi-relational graph convolution with position, velocity, and acceleration relations
- Improved adjacency matrix construction with motion-aware features
- Temporal convolution networks for sequence prediction

### Group-Aware Prediction
- Velocity-aware group detection
- Attention-based group integration
- Adaptive sampling for group-coherent predictions
- Intra-group and inter-group interaction modeling

### Comprehensive Loss Function
- Basic trajectory prediction loss
- Group consistency loss for coherent group behavior
- Velocity consistency loss for motion realism
- Social interaction loss for collision avoidance
- Trajectory smoothness loss for natural motion

## Architecture

```
Input Trajectories
       ↓
Enhanced DMRGCN (Velocity + Acceleration + Position Relations)
       ↓
Group-Aware Predictor (Automatic Group Detection)
       ↓
Feature Fusion (DMRGCN + Group Features)
       ↓
Final Prediction Head
       ↓
Multi-modal Trajectory Predictions
```

## Installation

```bash
# Clone the repository
cd plz/

# Install dependencies
pip install torch torchvision
pip install numpy matplotlib seaborn tqdm
pip install tensorboard scikit-learn
```

## Dataset Structure

```
dataset/
├── eth/
│   ├── train/
│   ├── val/
│   └── test/
├── hotel/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```

Each dataset file contains trajectory data in the format:
```
<frame_id> <ped_id> <x> <y>
```

## Usage

### Training

```bash
python train.py --dataset eth --tag unified-model-v1 \
                --num_epochs 300 --batch_size 64 \
                --lr 0.001 --visualize
```

Key training arguments:
- `--dataset`: Dataset name (eth, hotel, univ, zara1, zara2)
- `--tag`: Experiment tag for logging and checkpoints
- `--d_type`: Group detection type (velocity_aware, learned, euclidean)
- `--mix_type`: Group integration method (attention, mlp, cnn)
- `--include_velocity`: Include velocity features
- `--include_acceleration`: Include acceleration features

### Testing

```bash
python test.py --tag unified-model-v1 --dataset eth \
               --n_samples 20 --visualize
```

Testing arguments:
- `--n_samples`: Number of trajectory samples for evaluation
- `--n_trials`: Number of evaluation trials for robust metrics
- `--visualize`: Generate trajectory visualizations

### Model Configuration

The model can be configured with various architectural choices:

```python
model = UnifiedTrajectoryPredictor(
    n_stgcn=1,                    # Number of STGCN layers
    n_tpcnn=4,                    # Number of temporal CNN layers
    d_type='velocity_aware',      # Group detection method
    mix_type='attention',         # Group integration method
    group_type=(True, True, True) # (original, inter-group, intra-group)
)
```

## Results

### Quantitative Metrics
- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance at the final predicted time step
- **COL (Collision Rate)**: Percentage of predictions with collision
- **TCC (Trajectory Correlation Coefficient)**: Correlation between predicted and ground truth motion patterns

### Group-Specific Metrics
- **Group ADE/FDE**: Metrics computed specifically for group members
- **Group Coherence**: How well group members maintain coherent motion
- **Velocity Accuracy**: Accuracy of predicted velocities

## File Structure

```
plz/
├── model/
│   ├── __init__.py
│   ├── enhanced_dmrgcn.py      # Enhanced DMRGCN implementation
│   ├── group_aware_predictor.py # Group detection and integration
│   ├── unified_model.py        # Main unified model
│   └── loss.py                 # Comprehensive loss functions
├── utils/
│   ├── __init__.py
│   ├── dataloader.py          # Enhanced dataset loader
│   ├── metrics.py             # Evaluation metrics
│   └── visualizer.py          # Visualization tools
├── train.py                   # Training script
├── test.py                    # Testing script
├── checkpoints/               # Model checkpoints
└── dataset/                   # Dataset directory
```

## Key Innovations

1. **Velocity-Aware Group Detection**: Groups are formed based on both spatial proximity and velocity similarity
2. **Multi-Relational Adjacency**: Separate adjacency matrices for position, velocity, and acceleration relationships
3. **Attention-Based Integration**: Smart fusion of individual and group-level predictions
4. **Adaptive Sampling**: Group-aware sampling for realistic multi-modal predictions
5. **Comprehensive Loss**: Multi-objective optimization for realistic and socially-aware predictions

## Baseline Comparisons

The model builds upon and enhances:
- **DMRGCN**: By adding group awareness and velocity/acceleration modeling
- **GPGraph**: By incorporating enhanced motion dynamics and multi-relational graphs
- **Social-STGCN**: Through improved social interaction modeling
- **SGCN**: With better graph structure learning

## Citation

If you use this code in your research, please cite the original papers:

```bibtex
# DMRGCN Paper
@article{dmrgcn,
  title={Dynamic Multi-Relational Graph Convolutional Networks for Pedestrian Trajectory Prediction},
  author={...},
  journal={...},
  year={...}
}

# GPGraph Paper  
@article{gpgraph,
  title={Group-aware Pedestrian Graph for Multi-modal Trajectory Prediction},
  author={...},
  journal={...},
  year={...}
}
```

## License

This project is licensed under the MIT License - see the original project licenses for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Slow training**: Enable data parallel training or reduce model complexity
3. **Poor convergence**: Adjust learning rate, loss weights, or add gradient clipping

### Performance Tips

1. Use mixed precision training for faster computation
2. Implement data parallel training for multiple GPUs
3. Use efficient data loading with multiple workers
4. Cache preprocessed data for faster loading

## Future Work

- [ ] Multi-scale temporal modeling
- [ ] Transformer-based architectures
- [ ] Real-time inference optimization
- [ ] Extended evaluation on more datasets
- [ ] Integration with robotic navigation systems
