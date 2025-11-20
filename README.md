<h2 align="center">Disentangled Multi-Relational Graph Convolutional Network for<br>Pedestrian Trajectory Prediction</h2>
<p align="center">
  <a href="https://InhwanBae.github.io/"><strong>Inhwan Bae</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ"><strong>Hae-Gon Jeon</strong></a>
  <br>
  AAAI 2021
</p>

<p align="center">
  <a href="https://inhwanbae.github.io/publication/dmrgcn/"><strong><code>Project Page</code></strong></a>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16174"><strong><code>AAAI Paper</code></strong></a>
  <a href="https://github.com/InhwanBae/DMRGCN"><strong><code>Source Code</code></strong></a>
  <a href="#-citation"><strong><code>Related Works</code></strong></a>
</p>

<div align='center'>
  <br>
  <img src="img/stgcnn-probability-animated.webp" width=40%>
  &emsp;&emsp;
  <img src="img/dmrgcn-probability-animated.webp" width=40%>
</div>
<div align='center'>
  <span style='display:inline-block; width:40%; text-align:center'>Left: Previous SOTA Model (CVPR'20)</span>
  &emsp;&emsp;
  <span style='display:inline-block; width:40%; text-align:center'>Right: <b>DMRGCN (Ours)</b></span>
</div>

<!--<br>This repository contains the code for disentangling social interaction and alleviating accumulated errors for trajectory prediction.-->
<br>**Summary**: **Disentangling social interaction** and **alleviating accumulated errors** for trajectory prediction.

<br>

## 🧶 DMRGCN Model 🧶
* Addressing over-smoothing and biased weighting problems in high-order social relations.
* Disentangled Multi-scale Aggregation for better social interaction representation on a weighted graph.
* Global Temporal Aggregation for alleviating accumulated errors when pedestrians change their directions.
* DropEdge technique to avoid the over-fitting issue by randomly removing relation edges.


## Model Training
### Setup
**Environment**
<br>All models were trained and tested on Ubuntu 18.04 with Python 3.7 and PyTorch 1.6.0 with CUDA 10.1.

**Dataset**
<br>Preprocessed [ETH](https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz) and [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) datasets are included in this repository, under `./dataset/`. 
The train/validation/test splits are the same as those found in [Social-GAN](https://github.com/agrimgupta92/sgan).

### Train DMRGCN
To train our DMRGCN on the ETH and UCY datasets at once, we provide a bash script `train.sh` for a simplified execution.
```bash
./scripts/train.sh
```
We provide additional arguments for experiments: 
```bash
./scripts/train.sh <gpu_ids_for_five_scenes>

# Examples
./scripts/train.sh
./scripts/train.sh 0 0 0 0 0
./scripts/train.sh 0 1 2 3 4
```
If you want to train the model with custom hyper-parameters, use `train.py` instead of the script file.
```bash
python train.py --input_size <input_coordinate_dimension> --output_size <output_gaussian_dimension> \
--n_stgcn <number_of_gcn_layers> --n_tpcnn <number_of_cnn_layers> --kernel_size <kernel_size> \
--obs_seq_len <observation_sequence_length> --pred_seq_len <prediction_sequence_length> --dataset <dataset_name> \
--batch_size <minibatch_size> --num_epochs <number_of_epochs> --clip_grad <gradient_clipping> \
--lr <learning_rate> --lr_sh_rate <number_of_steps_to_drop_lr> --use_lrschd <use_lr_scheduler> \
--tag <experiment_tag> --visualize <visualize_trajectory>
```


## Model Evaluation
### Pretrained Models
We have included pretrained models in the `./checkpoints/` folder.

### Evaluate DMRGCN
You can use `test.py` to evaluate our model. 
```bash
python test.py --tag <experiment_tag>

# Examples
python test.py --tag social-dmrgcn-eth-experiment_tp4_de80
python test.py --tag social-dmrgcn-hotel-experiment_tp4_de80
python test.py --tag social-dmrgcn-univ-experiment_tp4_de80
python test.py --tag social-dmrgcn-zara1-experiment_tp4_de80
python test.py --tag social-dmrgcn-zara2-experiment_tp4_de80
```

#### Using Preprocessed Cache Files (Modified)
To use preprocessed `.pt` cache files instead of raw text files:
```bash
python test.py --tag <experiment_tag> --use_cache \
  --test_cache <path_to_test_cache_file> \
  --n_samples 20

# Example with SDD bookstore dataset
python test.py --tag sdd-bookstore-cached --use_cache \
  --test_cache /raid/guest/SDD_2beon/sdd_bookstore/test/preproc_cache_obs8_pred12_skip1.pt \
  --n_samples 20
```

**test.py Modifications Summary:**
- **Added `--use_cache` flag**: Enable use of preprocessed `.pt` cache files
- **Added `--test_cache` argument**: Path to test cache file (auto-detected if not specified)
- **Data loading logic** (lines 34-51): Conditional loading of `CachedTrajectoryDataset` or `TrajectoryDataset` based on `--use_cache` flag
- **Dimension handling** (lines 89-108): 
  - Fixed dimension mismatch issue: `generate_statistics_matrices` expects 4D tensor `(batch, num_peds, seq_len, 5)`
  - Convert `V_pred` from `(seq_len, num_peds, 5)` to `(1, num_peds, seq_len, 5)` before calling `generate_statistics_matrices`
  - Restore batch dimension after processing
  - Adjust `V_pred_sample` dimension order: `(KSTEPS, num_peds, seq_len, 2)` → `(KSTEPS, seq_len, num_peds, 2)` to match original code
- **Trajectory conversion** (lines 110-123):
  - Use `V_obs_traj[-1, :, :]` to get last observation timestep coordinates
  - Properly broadcast dimensions for adding relative trajectories to absolute coordinates
- **Dataset path**: Changed from `./datasets/` to `./datasets_pedestrian/` to match training setup

**Note**: All original test logic remains unchanged. Only cache file support and dimension handling fixes were added.


## 📖 Citation
If you find this code useful for your research, please cite our trajectory prediction papers :)

[**`🏢🚶‍♂️ CrowdES (CVPR'25) 🏃‍♀️🏠`**](https://github.com/InhwanBae/Crowd-Behavior-Generation) **|**
[**`💭 VLMTrajectory (TPAMI) 💭`**](https://github.com/InhwanBae/LMTrajectory) **|**
[**`💬 LMTrajectory (CVPR'24) 🗨️`**](https://github.com/InhwanBae/LMTrajectory) **|**
[**`1️⃣ SingularTrajectory (CVPR'24) 1️⃣`**](https://github.com/InhwanBae/SingularTrajectory) **|**
[**`🌌 EigenTrajectory (ICCV'23) 🌌`**](https://github.com/InhwanBae/EigenTrajectory) **|** 
[**`🚩 Graph‑TERN (AAAI'23) 🚩`**](https://github.com/InhwanBae/GraphTERN) **|**
[**`🧑‍🤝‍🧑 GP‑Graph (ECCV'22) 🧑‍🤝‍🧑`**](https://github.com/InhwanBae/GPGraph) **|**
[**`🎲 NPSN (CVPR'22) 🎲`**](https://github.com/InhwanBae/NPSN) **|**
[**`🧶 DMRGCN (AAAI'21) 🧶`**](https://github.com/InhwanBae/DMRGCN)

```bibtex
@article{bae2021dmrgcn,
  title={Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```
<details open>
  <summary>More Information (Click to expand)</summary>

```bibtex
@inproceedings{bae2025crowdes,
  title={Continuous Locomotive Crowd Behavior Generation},
  author={Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

@article{bae2025vlmtrajectory,
  title={Social Reasoning-Aware Trajectory Prediction via Multimodal Language Model},
  author={Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}

@inproceedings{bae2024lmtrajectory,
  title={Can Language Beat Numerical Regression? Language-Based Multimodal Trajectory Prediction},
  author={Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{bae2024singulartrajectory,
  title={SingularTrajectory: Universal Trajectory Predictor Using Diffusion Model},
  author={Bae, Inhwan and Park, Young-Jae and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{bae2023eigentrajectory,
  title={EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting},
  author={Bae, Inhwan and Oh, Jean and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@article{bae2023graphtern,
  title={A Set of Control Points Conditioned Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

@inproceedings{bae2022gpgraph,
  title={Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}

@inproceedings{bae2022npsn,
  title={Non-Probability Sampling Network for Stochastic Human Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
</details>


### Acknowledgement
Part of our code is borrowed from [Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN). 
We thank the authors for releasing their code and models.
