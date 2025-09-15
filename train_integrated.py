"""
통합 DMRGCN + GP-Graph 모델 훈련 스크립트

이 스크립트는 DMRGCN과 GP-Graph를 통합한 모델을 훈련합니다.
원본 DMRGCN 훈련 스크립트를 기반으로 하되, GP-Graph 모듈과 추가 특징들을 통합했습니다.

원본 출처:
- DMRGCN: https://github.com/InhwanBae/DMRGCN (AAAI 2021)
- GP-Graph: https://github.com/InhwanBae/GPGraph (ECCV 2022)

통합 방식:
1. DMRGCN의 기본 훈련 구조 유지
2. GP-Graph 모듈 추가 (GroupGenerator, GroupIntegrator)
3. Local Density와 Group Size 특징 추가 (미팅에서 요청된 기능)
4. 모든 특징을 통합하여 최종 예측 수행
"""

import os
import pickle
import argparse
import torch
from tqdm import tqdm
from model import create_integrated_model, multivariate_loss  # 통합 모델과 손실 함수
from utils import TrajectoryDataset, data_sampler  # DMRGCN의 데이터 로더와 증강 함수
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# CUDA 관련 문제 방지 (원본 DMRGCN에서 가져옴)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# 명령행 인자 파싱
parser = argparse.ArgumentParser()

# DMRGCN 모델 파라미터 (원본 DMRGCN에서 가져옴)
parser.add_argument('--input_size', type=int, default=2)  # 입력 좌표 차원 (x, y)
parser.add_argument('--output_size', type=int, default=5)  # 출력 가우시안 차원 (μx, μy, σx, σy, ρ)
parser.add_argument('--n_stgcn', type=int, default=1, help='Number of GCN layers')
parser.add_argument('--n_tpcnn', type=int, default=4, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

# 데이터 관련 파라미터 (원본 DMRGCN에서 가져옴)
parser.add_argument('--obs_seq_len', type=int, default=8)  # 관찰 시퀀스 길이
parser.add_argument('--pred_seq_len', type=int, default=12)  # 예측 시퀀스 길이
parser.add_argument('--dataset', default='eth', help='Dataset name(eth,hotel,univ,zara1,zara2)')

# 훈련 관련 파라미터 (원본 DMRGCN에서 가져옴)
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=128, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=32, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='integrated-dmrgcn-gpgraph', help='Personal tag for the model')
parser.add_argument('--visualize', action="store_true", default=False, help='Visualize trajectories')

# GP-Graph 관련 파라미터 (GP-Graph에서 가져옴)
parser.add_argument('--group_d_type', default='learned_l2norm', help='Group distance type')
parser.add_argument('--group_d_th', default='learned', help='Group distance threshold')
parser.add_argument('--group_mix_type', default='mlp', help='Group mixing type')
parser.add_argument('--use_group_processing', action="store_true", default=True, help='Use group processing')

# 밀도/그룹 크기 파라미터 (미팅에서 요청된 새로운 기능)
parser.add_argument('--density_radius', type=float, default=2.0, help='Density computation radius')
parser.add_argument('--group_size_threshold', type=int, default=2, help='Group size threshold')
parser.add_argument('--use_density', action="store_true", default=True, help='Use local density feature')
parser.add_argument('--use_group_size', action="store_true", default=True, help='Use group size feature')

args = parser.parse_args()

# Data preparation
dataset_path = './datasets/' + args.dataset + '/'
checkpoint_dir = './checkpoints/' + args.tag + '/'

train_dataset = TrajectoryDataset(dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = TrajectoryDataset(dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = create_integrated_model(
    n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn, output_feat=args.output_size,
    kernel_size=args.kernel_size, seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len,
    group_d_type=args.group_d_type, group_d_th=args.group_d_th, group_mix_type=args.group_mix_type,
    use_group_processing=args.use_group_processing, density_radius=args.density_radius,
    group_size_threshold=args.group_size_threshold, use_density=args.use_density,
    use_group_size=args.use_group_size
)
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
    from utils import data_visualizer
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

        # 통합 모델에서 GP-Graph 그룹핑은 배치=1을 가정하므로 배치 증강 비활성화
        # 필요시 아래에서 batch=1로 유지
        aug = False
        if aug:
            V_obs, A_obs, V_tr, A_tr = data_sampler(V_obs, A_obs, V_tr, A_tr, batch=1)

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, group_indices = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        loss = multivariate_loss(V_pred, V_tr, training=True)
        loss.backward()
        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            iter_idx = epoch * loader_len + batch_idx
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
        V_pred, group_indices = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        loss = multivariate_loss(V_pred, V_tr)
        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            # Visualize trajectories
            if args.visualize:
                fig_img = data_visualizer(V_pred, obs_traj, pred_traj_gt, samples=100)
                writer.add_image('Valid_{0:04d}'.format(batch_idx), fig_img[:, :, :], epoch, dataformats='HWC')

            iter_idx = epoch * loader_len + batch_idx
            writer.add_scalar('Loss/Valid_V', (loss_batch / batch_idx), iter_idx)

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
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
        print("Train_loss: {0}, Val_loss: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
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
