# DMRGCN + GP-Graph Integration

This project integrates DMRGCN (Disentangled Multi-Relational Graph Convolution Network) with GP-Graph (Group-aware Pedestrian Graph) for enhanced trajectory prediction with group-aware modeling.

## 🎯 Overview

The integration combines:
- **DMRGCN backbone**: Multi-relational graph processing with disentangled adjacency matrices
- **GP-Graph modules**: Group assignment and hierarchical processing (agent-level, intra-group, inter-group)
- **Shared architecture**: DMRGCN backbone shared across three processing paths
- **Feature fusion**: Advanced fusion strategies for combining multi-path features

## 🏗️ Architecture

### Forward Path
1. **Group Assignment**: Assign pedestrians to groups based on spatial proximity
2. **Graph Construction**: Build three types of graphs:
   - Agent-level (individual pedestrian interactions)
   - Intra-group (within-group interactions)
   - Inter-group (between-group interactions)
3. **Parallel Processing**: Process each graph type with shared DMRGCN backbone
4. **Feature Fusion**: Combine features from all paths
5. **Prediction**: Generate Gaussian trajectory parameters

### Key Components

#### Datasets (`datasets/`)
- Enhanced dataloader with agent ID preservation
- Pairwise distance and displacement matrix computation
- Custom collate function for variable-sized sequences

#### Utils (`utils/`)
- Graph building utilities
- Group pooling/unpooling operations
- Multi-relational adjacency construction

#### Model (`model/`)
- **backbone.py**: Modular DMRGCN blocks
- **gpgraph_adapter.py**: Group assignment and integration modules
- **dmrgcn_gpgraph.py**: Main integrated model
- **utils.py**: Model utilities (normalization, edge dropping, etc.)

## 🚀 Usage

### Training

Basic training with default group-aware settings:
```bash
python train.py --dataset eth --gpgraph --group_type euclidean --group_th 2.0
```

Advanced training with learned grouping:
```bash
python train.py \\
    --dataset hotel \\
    --gpgraph \\
    --group_type learned \\
    --group_th 1.5 \\
    --mix_type mlp \\
    --st_estimator \\
    --batch_size 32 \\
    --num_epochs 128
```

### Testing

Evaluate model with motion analysis:
```bash
python test.py \\
    --dataset eth \\
    --checkpoint ./checkpoints/model_best.pth \\
    --motion_analysis \\
    --visualize \\
    --num_samples 20
```

### Batch Scripts

Use provided shell scripts for automated training/testing:
```bash
# Training
bash run_train.sh

# Testing  
bash run_test.sh
```

## 📊 Key Parameters

### Group-Aware Settings
- `--group_type`: Group assignment method (`euclidean`, `learned`, `learned_l2norm`, `estimate_th`)
- `--group_th`: Distance threshold for grouping
- `--mix_type`: Feature fusion method (`mean`, `sum`, `mlp`, `cnn`, `attention`)
- `--enable_agent/intra/inter`: Enable specific processing paths
- `--st_estimator`: Use spatio-temporal features for group estimation

### DMRGCN Settings
- `--distance_scales`: Distance scales for multi-relational graphs
- `--hidden_dims`: Hidden dimensions for backbone network
- `--share_backbone`: Share backbone across processing paths
- `--use_mdn`: Use mixture density network for predictions

## 🔧 Architecture Details

### Dimension Management
Careful attention to dimension consistency:
- Agent-level: `(N, T, F)` 
- Group-level: `(Ng, T, F)` where `Ng` is number of groups
- Adjacency: `(B, R, T, N, N)` for agent-level, `(B, R, T, Ng, Ng)` for group-level
- Unpool operation restores `(N, T, F)` from group features

### Loss Function
Negative log-likelihood for Gaussian predictions:
- Predicted parameters: `[μ_x, μ_y, log_σ_x, log_σ_y, ρ]`
- Multivariate normal distribution
- Numerical stability with diagonal regularization

## 📈 Evaluation Metrics

### Standard Metrics
- **ADE** (Average Displacement Error): Average L2 distance over all predicted points
- **FDE** (Final Displacement Error): L2 distance at final predicted point

### Motion Subset Analysis
Pedestrians categorized by motion patterns:
- **Static**: Low velocity, low acceleration
- **Linear**: High velocity, low acceleration  
- **Non-linear**: High velocity, high acceleration
- **Accelerating**: Low velocity, high acceleration

### Group Analysis
- Group size distribution
- Intra-group vs inter-group interaction analysis
- Group-specific prediction accuracy

## 🎯 Key Features

### Innovation Points
1. **Shared Backbone**: DMRGCN backbone shared across three processing paths
2. **Hierarchical Processing**: Agent → Intra-group → Inter-group features
3. **Flexible Grouping**: Multiple group assignment strategies
4. **Advanced Fusion**: Multiple feature integration methods
5. **Motion Analysis**: Detailed motion pattern evaluation

### Technical Improvements
- Device-agnostic implementation
- Numerical stability enhancements
- Efficient batch processing
- Comprehensive error handling

## 📁 Project Structure

```
.
├── datasets/
│   ├── __init__.py
│   └── dataloader.py          # Enhanced dataset with group features
├── model/
│   ├── __init__.py
│   ├── backbone.py            # Modular DMRGCN backbone
│   ├── gpgraph_adapter.py     # Group assignment & integration
│   ├── dmrgcn_gpgraph.py      # Main integrated model
│   └── utils.py               # Model utilities
├── utils/
│   ├── __init__.py
│   └── graph_build.py         # Graph construction utilities
├── train.py                   # Training script
├── test.py                    # Testing script
├── run_train.sh              # Training automation
├── run_test.sh               # Testing automation
├── requirements.txt          # Dependencies
└── README_INTEGRATION.md     # This file
```

## 🔍 Troubleshooting

### Common Issues
1. **Dimension Mismatch**: Check tensor shapes during pooling/unpooling
2. **GPU Memory**: Reduce batch size or sequence length
3. **Numerical Instability**: Check covariance matrix conditioning
4. **Group Assignment**: Verify distance threshold and grouping method

### Performance Tips
- Use `--share_backbone` for memory efficiency
- Adjust `--group_th` based on dataset density
- Use `--mix_type mean` for stable training
- Enable `--st_estimator` for complex scenes

## 📚 References

1. DMRGCN: Disentangled Multi-Relational Graph Convolution Network
2. GP-Graph: Group-aware Pedestrian Graph for Trajectory Prediction
3. Social-GAN: Socially Acceptable Trajectories with Generative Adversarial Networks
