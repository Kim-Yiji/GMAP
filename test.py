import pickle
import argparse
import torch
import os

from tqdm import tqdm
from utils import TrajectoryDataset, CachedTrajectoryDataset, SDDTrajectoryDataset
from model import *

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import multivariate_normal

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
parser.add_argument('--visualize', action="store_true", default=False, help='Visualize trajectories')
parser.add_argument('--use_cache', action="store_true", default=False, help='Use preprocessed .pt cache files instead of raw text files')
parser.add_argument('--test_cache', default=None, help='Path to test cache file (default: auto-detect in dataset_path/test/)')
test_args = parser.parse_args()

# Get arguments for training
checkpoint_dir = './checkpoints/' + test_args.tag + '/'

args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

# Check if dataset_type exists in args (for backward compatibility)
dataset_type = getattr(args, 'dataset_type', 'ethucy')

# Model path
if dataset_type == 'sdd':
    model_path = checkpoint_dir + args.dataset + '.pth'
else:
    model_path = checkpoint_dir + args.dataset + '_best.pth'
KSTEPS = test_args.n_samples

# Data preparation
if dataset_type == 'sdd':
    # SDD dataset
    if test_args.use_cache:
        raise ValueError("use_cache is not supported for dataset_type='sdd'")
    
    sdd_base = '/raid/guest/OATMeal_Queens/SDD_datasets'
    test_dir = os.path.join(sdd_base, f'sdd_{args.dataset}', 'test')
    
    if not os.path.isdir(test_dir):
        fallback_dir = os.path.join(sdd_base, 'SDD_raw', args.dataset, 'video0')
        if not os.path.isdir(fallback_dir):
            raise FileNotFoundError(f"SDD test directory not found: {test_dir} (fallback {fallback_dir} missing)")
        print(f"Using fallback SDD dataset from: {fallback_dir}")
        test_dataset = SDDTrajectoryDataset(fallback_dir, obs_len=args.obs_seq_len, pred_len=args.pred_seq_len)
    else:
        print(f"Using SDD dataset from: {test_dir}")
        test_dataset = SDDTrajectoryDataset(test_dir, obs_len=args.obs_seq_len, pred_len=args.pred_seq_len)
elif test_args.use_cache:
    # Use preprocessed .pt cache files
    import glob
    
    dataset_path = './datasets_pedestrian/' + args.dataset + '/'
    # Auto-detect cache file if not specified
    if test_args.test_cache is None:
        test_cache_files = glob.glob(dataset_path + 'test/*.pt')
        if test_cache_files:
            test_args.test_cache = test_cache_files[0]
            print(f"Auto-detected test cache: {test_args.test_cache}")
        else:
            raise ValueError(f"No .pt cache file found in {dataset_path}test/")
    
    test_dataset = CachedTrajectoryDataset(test_args.test_cache)
else:
    # Use raw text files (original behavior)
    dataset_path = './datasets_pedestrian/' + args.dataset + '/'
    test_dataset = TrajectoryDataset(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = social_dmrgcn(n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn,
                      output_feat=args.output_size, kernel_size=args.kernel_size,
                      seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len)
model = model.cuda()
model.load_state_dict(torch.load(model_path))

# Test logging
writer = SummaryWriter(checkpoint_dir)
if test_args.visualize:
    from utils import data_visualizer


def test(KSTEPS=20):
    model.eval()

    ade_all = []
    fde_all = []

    progressbar = tqdm(range(len(test_loader)))
    progressbar.set_description('Testing {}'.format(test_args.tag))

    for batch_idx, batch in enumerate(test_loader):
        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
        obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)  # (batch=1, seq_len, num_peds, 5)

        # 원본 코드 유지: V_pred.squeeze() 후 처리
        V_pred = V_pred.squeeze()
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)  # (obs_len, num_peds, 2)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)  # (pred_len, num_peds, 2)

        # Randomly sampling predict trajectories
        # 원본 코드: generate_statistics_matrices(V_pred.squeeze(dim=0))
        # 하지만 generate_statistics_matrices는 4차원을 기대하므로 조정 필요
        # V_pred는 (seq_len, num_peds, 5) 형태일 것
        # generate_statistics_matrices에 맞게 (1, num_peds, seq_len, 5)로 변환
        if V_pred.dim() == 3:
            # (seq_len, num_peds, 5) -> (1, num_peds, seq_len, 5)
            V_pred_for_stats = V_pred.permute(1, 0, 2).unsqueeze(0)
        else:
            V_pred_for_stats = V_pred.unsqueeze(0)
        
        mu, cov = generate_statistics_matrices(V_pred_for_stats)
        mu = mu.squeeze(dim=0)  # (num_peds, seq_len, 2)
        cov = cov.squeeze(dim=0)  # (num_peds, seq_len, 2, 2)
        
        mv_normal = multivariate_normal.MultivariateNormal(mu, cov)
        V_pred_sample = mv_normal.sample((KSTEPS,))  # (KSTEPS, num_peds, seq_len, 2)
        
        # 원본 코드는 V_pred_sample.size(1)을 사용하므로 (KSTEPS, seq_len, num_peds, 2) 형태를 기대
        V_pred_sample = V_pred_sample.permute(0, 2, 1, 3)  # (KSTEPS, seq_len, num_peds, 2)

        # Relative trajectories to absolute trajectories
        # 원본 코드: V_obs_traj[-1, :, :] 사용
        # V_obs_traj: (obs_len, num_peds, 2) 형태
        # V_obs_traj[-1, :, :]: (num_peds, 2) - 마지막 타임스텝의 모든 pedestrian 좌표
        V_absl = []
        obs_last = V_obs_traj[-1, :, :]  # (num_peds, 2)
        obs_last = obs_last.unsqueeze(0).unsqueeze(0)  # (1, 1, num_peds, 2)
        
        for t in range(V_pred_sample.size(1)):  # seq_len
            # V_pred_sample[:, 0:t + 1, :, :]: (KSTEPS, t+1, num_peds, 2)
            # .sum(dim=1, keepdim=True): (KSTEPS, 1, num_peds, 2)
            sum_result = V_pred_sample[:, 0:t + 1, :, :].sum(dim=1, keepdim=True)  # (KSTEPS, 1, num_peds, 2)
            V_absl.append(sum_result + obs_last)
        V_absl = torch.cat(V_absl, dim=1)

        # Calculate ADEs and FDEs for each trajectory
        temp = V_absl - V_pred_traj_gt
        temp = (temp ** 2).sum(dim=-1).sqrt()

        ADEs = temp.mean(dim=1).min(dim=0)[0]
        FDEs = temp[:, -1, :].min(dim=0)[0]

        ade_all.extend(ADEs.tolist())
        fde_all.extend(FDEs.tolist())

        # Visualize trajectories
        if test_args.visualize and batch_idx % 1 == 0:
            fig_img = data_visualizer(V_pred.unsqueeze(dim=0), obs_traj, pred_traj_gt, samples=100)
            writer.add_image('Test', fig_img[:, :, :], batch_idx, dataformats='HWC')

        progressbar.update(1)

    progressbar.close()

    ade_ = sum(ade_all) / len(ade_all)
    fde_ = sum(fde_all) / len(fde_all)

    return ade_, fde_


def main():
    ade, fde = test(KSTEPS)

    result_lines = ["Evaluating model: {}".format(test_args.tag),
                    "ADE: {0}, FDE: {1}".format(ade, fde)]

    with open(checkpoint_dir + 'results.txt', 'a') as f:
        for line in result_lines:
            f.write(line + '\n')
            print(line)


if __name__ == "__main__":
    main()

writer.close()
