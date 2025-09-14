"""
Testing script for integrated DMRGCN + GP-Graph model
"""

import os
import pickle
import argparse
import torch
import numpy as np
from model import create_integrated_model, multivariate_loss
from utils import TrajectoryDataset
from torch.utils.data import DataLoader

# To avoid contiguous problem.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='integrated-dmrgcn-gpgraph', help='Model tag')
parser.add_argument('--dataset', default='eth', help='Dataset name(eth,hotel,univ,zara1,zara2)')
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcn', type=int, default=1)
parser.add_argument('--n_tpcnn', type=int, default=4)
parser.add_argument('--kernel_size', type=int, default=3)

# GP-Graph specific parameters
parser.add_argument('--group_d_type', default='learned_l2norm')
parser.add_argument('--group_d_th', default='learned')
parser.add_argument('--group_mix_type', default='mlp')
parser.add_argument('--use_group_processing', action="store_true", default=True)

# Density/Group size parameters
parser.add_argument('--density_radius', type=float, default=2.0)
parser.add_argument('--group_size_threshold', type=int, default=2)
parser.add_argument('--use_density', action="store_true", default=True)
parser.add_argument('--use_group_size', action="store_true", default=True)

args = parser.parse_args()

# Load model arguments
checkpoint_dir = './checkpoints/' + args.tag + '/'
with open(checkpoint_dir + 'args.pkl', 'rb') as f:
    saved_args = pickle.load(f)

# Override with command line arguments
for key, value in vars(args).items():
    if hasattr(saved_args, key):
        setattr(saved_args, key, value)

# Data preparation
dataset_path = './datasets/' + args.dataset + '/'
test_dataset = TrajectoryDataset(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = create_integrated_model(
    n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn, output_feat=args.output_size,
    kernel_size=args.kernel_size, seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len,
    group_d_type=args.group_d_type, group_d_th=args.group_d_th, group_mix_type=args.group_mix_type,
    use_group_processing=args.use_group_processing, density_radius=args.density_radius,
    group_size_threshold=args.group_size_threshold, use_density=args.use_density,
    use_group_size=args.use_group_size
)

# Load model weights
model_path = checkpoint_dir + args.dataset + '_best.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
else:
    print(f"Model file not found: {model_path}")
    exit(1)

model = model.cuda()
model.eval()

# Test function
def test():
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
            
            V_obs_ = V_obs.permute(0, 3, 1, 2)
            V_pred, group_indices = model(V_obs_, A_obs)
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            loss = multivariate_loss(V_pred, V_tr)
            total_loss += loss.item()
            total_samples += 1
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / total_samples
    print(f"Average Test Loss: {avg_loss:.6f}")
    return avg_loss

if __name__ == "__main__":
    print("Testing integrated DMRGCN + GP-Graph model...")
    print(f"Dataset: {args.dataset}")
    print(f"Model tag: {args.tag}")
    print(f"Group processing: {args.use_group_processing}")
    print(f"Density feature: {args.use_density}")
    print(f"Group size feature: {args.use_group_size}")
    print("-" * 50)
    
    test_loss = test()
    print(f"Final test loss: {test_loss:.6f}")
