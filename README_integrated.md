# Integrated DMRGCN + GP-Graph Model

This repository contains an integrated model combining DMRGCN (Disentangled Multi-Relational Graph Convolutional Network) and GP-Graph (Learning Pedestrian Group Representations) for pedestrian trajectory prediction.

## Features

### Core Integration
- **DMRGCN Base**: Multi-relational graph convolution with disentangled aggregation
- **GP-Graph Modules**: Group-based processing with pedestrian group pooling/unpooling
- **Feature Integration**: Local density and group size features

### Additional Features
- **Local Density**: Computes pedestrian density around each individual
- **Group Size**: Tracks group membership and size
- **Motion Analysis**: Analyzes different types of pedestrian motions
- **Filtered Datasets**: Creates motion-specific datasets for targeted evaluation

## Model Architecture

```
Input Trajectory
    ↓
GP-Graph Group Generation (GroupGenerator)
    ↓
DMRGCN Multi-Relational Processing
    - Original Graph (Individual interactions)
    - Inter-group Graph (Group-level interactions) 
    - Intra-group Graph (Within-group interactions)
    ↓
Feature Integration (Density + Group Size)
    ↓
Group Integration (GroupIntegrator)
    ↓
Output Prediction
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib tqdm tensorboard
```

## Usage

### Training the Integrated Model

```bash
python train_integrated.py --dataset eth --tag integrated-dmrgcn-gpgraph --use_group_processing --use_density --use_group_size
```

### Testing the Model

```bash
python test_integrated.py --tag integrated-dmrgcn-gpgraph --dataset eth
```

### Motion Analysis

Analyze different types of motions in the dataset:

```bash
python analyze_motions.py --dataset eth --data_split train --save_plots
```

### Create Filtered Datasets

Create motion-specific datasets:

```bash
python create_filtered_dataset.py --source_dataset eth --create_all_combinations
```

### Performance Comparison

Compare model performance across different datasets:

```bash
python compare_performance.py --model_tag integrated-dmrgcn-gpgraph --dataset eth --compare_filtered
```

## Model Parameters

### DMRGCN Parameters
- `n_stgcn`: Number of GCN layers (default: 1)
- `n_tpcnn`: Number of CNN layers (default: 4)
- `kernel_size`: Kernel size (default: 3)
- `output_feat`: Output feature dimension (default: 5)

### GP-Graph Parameters
- `group_d_type`: Group distance type ('learned_l2norm', 'learned', 'euclidean')
- `group_d_th`: Group distance threshold ('learned', float)
- `group_mix_type`: Group mixing type ('mlp', 'cnn', 'mean', 'sum')
- `use_group_processing`: Enable group-based processing

### Feature Parameters
- `density_radius`: Radius for density computation (default: 2.0)
- `group_size_threshold`: Minimum group size threshold (default: 2)
- `use_density`: Enable local density feature
- `use_group_size`: Enable group size feature

## Dataset Structure

The model works with ETH/UCY datasets in the following format:
```
datasets/
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

## Motion Types

The model can handle and analyze different types of pedestrian motions:

1. **Linear Motion**: Straight-line trajectories with minimal direction changes
2. **Curved Motion**: Non-linear trajectories with smooth curves
3. **Direction Change**: Trajectories with significant direction changes
4. **Group Motion**: High-speed, coordinated group movements
5. **Stationary**: Minimal movement trajectories

## Performance Metrics

- **ADE (Average Displacement Error)**: Average error across all time steps
- **FDE (Final Displacement Error)**: Error at the final prediction time step
- **Loss**: Multivariate Gaussian loss for trajectory prediction

## File Structure

```
├── model/
│   ├── __init__.py
│   ├── dmrgcn.py                    # Original DMRGCN implementation
│   ├── gpgraph_modules.py          # GP-Graph modularized components
│   ├── social_dmrgcn_gpgraph.py    # Integrated model
│   └── ...                         # Other model components
├── train_integrated.py             # Training script
├── test_integrated.py              # Testing script
├── analyze_motions.py              # Motion analysis script
├── create_filtered_dataset.py      # Dataset filtering script
├── compare_performance.py          # Performance comparison script
└── README_integrated.md            # This file
```

## Example Results

After training and evaluation, you can expect to see:

1. **Motion Analysis**: Distribution of different motion types in the dataset
2. **Performance Comparison**: ADE/FDE metrics across different datasets
3. **Feature Impact**: Effect of density and group size features on performance

## Citation

If you use this integrated model, please cite both original papers:

```bibtex
@article{bae2021dmrgcn,
  title={Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}

@inproceedings{bae2022gpgraph,
  title={Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
