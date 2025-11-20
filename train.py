import os
import pickle
import argparse
import torch

from tqdm import tqdm
from model import *
from utils import TrajectoryDataset, CachedTrajectoryDataset, data_sampler

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
parser.add_argument('--dataset', default='eth', help='Dataset name(eth,hotel,univ,zara1,zara2)')

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
dataset_path = './datasets_pedestrian/' + args.dataset + '/'
checkpoint_dir = './checkpoints/' + args.tag + '/'

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
    skipped_batches = 0
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
        try:
            loss = multivariate_loss(V_pred, V_tr, training=True)
        except ValueError as err:
            skipped_batches += 1
            progressbar.write(f"[Epoch {epoch}] Skipping train batch {batch_idx}: {err}")
            continue

        # Skip batches with extremely large loss to prevent gradient explosion
        if loss.item() > 1000.0:
            skipped_batches += 1
            progressbar.write(f"[Epoch {epoch}] Skipping train batch {batch_idx}: Loss too large ({loss.item():.2e})")
            optimizer.zero_grad()
            continue
        
        loss.backward()
        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            # Always clip gradients to prevent explosion
            clip_value = args.clip_grad if args.clip_grad is not None else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            iter_idx = epoch * loader_len + batch_idx
            avg_train_loss = loss_batch / (batch_idx + 1)
            writer.add_scalar('Loss/Train_V', avg_train_loss, iter_idx)

        progressbar.set_description('Train Epoch: {0} Loss: {1:.6f}'.format(epoch, loss.item()))
        progressbar.update(1)

    progressbar.close()
    if skipped_batches:
        print(f"[Epoch {epoch}] Skipped {skipped_batches} training batches due to invalid loss.")

    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))
    skipped_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
        obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        try:
            loss = multivariate_loss(V_pred, V_tr)
        except ValueError as err:
            skipped_batches += 1
            progressbar.write(f"[Epoch {epoch}] Skipping valid batch {batch_idx}: {err}")
            continue

        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            # Visualize trajectories
            if args.visualize:
                fig_img = data_visualizer(V_pred, obs_traj, pred_traj_gt, samples=100)
                writer.add_image('Valid_{0:04d}'.format(batch_idx), fig_img[:, :, :], epoch, dataformats='HWC')

            iter_idx = epoch * loader_len + batch_idx
            avg_val_loss = loss_batch / (batch_idx + 1)
            writer.add_scalar('Loss/Valid_V', avg_val_loss, iter_idx)

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.6f}'.format(epoch, loss.item()))
        progressbar.update(1)

    progressbar.close()
    if skipped_batches:
        print(f"[Epoch {epoch}] Skipped {skipped_batches} validation batches due to invalid loss.")

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
