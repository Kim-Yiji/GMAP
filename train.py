import os
import pickle
import argparse
import torch

from tqdm import tqdm
from model import *
from utils import TrajectoryDataset, CachedTrajectoryDataset, SDDTrajectoryDataset, data_sampler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# To avoid contiguous problem.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcn', type=int, default=1, help='Number of GCN layers')
parser.add_argument('--n_tpcnn', type=int, default=4, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

# Data specific parameters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='Dataset name(eth,hotel,univ,zara1,zara2 or SDD scene name)')
parser.add_argument('--dataset_type', default='ethucy', choices=['ethucy', 'sdd'],
                    help='Data format type: ethucy (original) or sdd (Stanford Drone Dataset)')

# Training specific parameters
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=128, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=32, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--visualize', action="store_true", default=False, help='Visualize trajectories')
parser.add_argument('--use_cache', action="store_true", default=False, help='Use preprocessed .pt cache files instead of raw text files')
parser.add_argument('--train_cache', default=None, help='Path to train cache file (default: auto-detect in dataset_path/train/)')
parser.add_argument('--val_cache', default=None, help='Path to val cache file (default: auto-detect in dataset_path/val/)')

args = parser.parse_args()

# Data preparation
# Batch size set to 1 because vertices vary by humans in each scene sequence.
# Use mini batch working like batch.
checkpoint_dir = './checkpoints/' + args.tag + '/'

if args.dataset_type == 'sdd':
    # SDD_datasets의 새로운 구조 사용
    sdd_base = '/raid/guest/OATMeal_Queens/SDD_datasets'
    scene_dir = os.path.join(sdd_base, f'sdd_{args.dataset}')
    
    # .pt 파일이 있으면 캐시 사용
    train_pt = os.path.join(scene_dir, 'train.pt')
    val_pt = os.path.join(scene_dir, 'val.pt')
    
    if os.path.exists(train_pt) and os.path.exists(val_pt):
        print(f"Using preprocessed .pt files:")
        print(f"  Train: {train_pt}")
        print(f"  Val: {val_pt}")
        train_dataset = CachedTrajectoryDataset(train_pt)
        val_dataset = CachedTrajectoryDataset(val_pt)
    else:
        # .pt 파일이 없으면 CSV 파일에서 직접 로드
        train_dir = os.path.join(scene_dir, 'train')
        val_dir = os.path.join(scene_dir, 'val')
        
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"SDD train directory not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"SDD val directory not found: {val_dir}")
        
        print(f"Using SDD CSV files (will preprocess on-the-fly):")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
        train_dataset = SDDTrajectoryDataset(train_dir, obs_len=args.obs_seq_len, pred_len=args.pred_seq_len)
        val_dataset = SDDTrajectoryDataset(val_dir, obs_len=args.obs_seq_len, pred_len=args.pred_seq_len)
else:
    # ETH/UCY 형식 (기존 동작)
    dataset_path = './datasets_pedestrian/' + args.dataset + '/'

    if args.use_cache:
        # Use preprocessed .pt cache files
        import glob
        
        # Auto-detect cache files if not specified
        if args.train_cache is None:
            train_cache_files = glob.glob(dataset_path + 'train/*.pt')
            if train_cache_files:
                args.train_cache = train_cache_files[0]
                print(f"Auto-detected train cache: {args.train_cache}")
            else:
                raise ValueError(f"No .pt cache file found in {dataset_path}train/")
        
        if args.val_cache is None:
            val_cache_files = glob.glob(dataset_path + 'val/*.pt')
            if val_cache_files:
                args.val_cache = val_cache_files[0]
                print(f"Auto-detected val cache: {args.val_cache}")
            else:
                raise ValueError(f"No .pt cache file found in {dataset_path}val/")
        
        train_dataset = CachedTrajectoryDataset(args.train_cache)
        val_dataset = CachedTrajectoryDataset(args.val_cache)
    else:
        # Use raw text files (original behavior)
        train_dataset = TrajectoryDataset(dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
        val_dataset = TrajectoryDataset(dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = social_dmrgcn(n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn,
                      output_feat=args.output_size, kernel_size=args.kernel_size,
                      seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

writer = SummaryWriter(checkpoint_dir)
if args.visualize:
    try:
        from utils.visualizer import data_visualizer
    except ImportError:
        print("Warning: Visualizer not available, disabling visualization")
        args.visualize = False
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def train(epoch):
    global metrics
    model.train()
    loss_batch = 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(train_loader):
        # Sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]

        # Try augmentation to generate a batch.
        aug = True
        if aug:
            V_obs, A_obs, V_tr, A_tr = data_sampler(V_obs, A_obs, V_tr, A_tr, batch=4)

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        # 디버그: 텐서 차원 확인
        #print(f"DEBUG: V_pred.shape = {V_pred.shape}")
        #print(f"DEBUG: V_tr.shape = {V_tr.shape}")
        loss = multivariate_loss(V_pred, V_tr, training=True)
        loss.backward()
        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            iter_idx = epoch * loader_len + batch_idx
            # batch_idx가 0일 때는 평균을 계산할 수 없으므로 로그만 스킵
            if batch_idx > 0:
                writer.add_scalar('Loss/Train_V', (loss_batch / batch_idx), iter_idx)

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()

    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
        obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        loss = multivariate_loss(V_pred, V_tr)
        loss_value = loss.item()
        # NaN 체크: NaN이면 0으로 대체
        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            loss_value = 0.0
        loss_batch += loss_value

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            # Visualize trajectories
            if args.visualize:
                fig_img = data_visualizer(V_pred, obs_traj, pred_traj_gt, samples=100)
                writer.add_image('Valid_{0:04d}'.format(batch_idx), fig_img[:, :, :], epoch, dataformats='HWC')

            iter_idx = epoch * loader_len + batch_idx
            if batch_idx > 0:
                writer.add_scalar('Loss/Valid_V', (loss_batch / batch_idx), iter_idx)

        loss_display = loss.item()
        if torch.isnan(torch.tensor(loss_display)) or torch.isinf(torch.tensor(loss_display)):
            loss_display = 0.0
        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss_display / args.batch_size))
        progressbar.update(1)

    progressbar.close()

    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    torch.save(model.state_dict(), checkpoint_dir + args.dataset + '.pth')
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best.pth')


def main():
    for epoch in range(args.num_epochs):
        train(epoch)
        valid(epoch)
        if args.use_lrschd:
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'],
                                                             constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()

writer.close()
