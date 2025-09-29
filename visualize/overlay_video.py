"""
Overlay predicted human trajectories from the unified DMRGCN+GP-Graph model onto a video.

This script is designed to be instructional: it favors clarity over brevity and
explains each step with detailed comments. It supports two overlay modes:

1) snapshot: draw the entire future prediction on the last observed frame
2) temporal: draw the future step-by-step aligned with subsequent video frames

Requirements:
 - OpenCV (cv2) for reading/writing video and drawing
 - PyTorch for model execution

Typical usage (snapshot mode with simple scaling instead of homography):
  python visualize/overlay_video.py \
    --video path/to/scene.mp4 \
    --output output/overlay_demo.mp4 \
    --mode snapshot \
    --scale 20 20 --offset 0 0 --flip-y

If you have a homography matrix H (scene/world -> image pixels), save it as a
npz file with key 'H' and pass --homography path/to/H.npz (this overrides
--scale/--offset/--flip-y).

Note: This script fetches one batch from your dataset, runs the model to obtain
      predicted absolute positions, and overlays them. You may adapt the data
      alignment (frame indices) as needed per dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2
import torch

# Ensure project root is on sys.path when running this file directly
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import your model and dataloader. Adjust paths if your package layout differs.
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from datasets.dataloader import TrajectoryDataset, collate_fn
from torch.utils.data import DataLoader


# ----------------------------
# Coordinate mapping utilities
# ----------------------------

def load_homography(h_path: str) -> np.ndarray:
    """Load a 3x3 homography matrix from npz or txt.

    The npz file should contain key 'H'. For txt, it expects 9 whitespace
    separated values (row-major).
    """
    ext = os.path.splitext(h_path)[1].lower()
    if ext == '.npz':
        data = np.load(h_path)
        if 'H' not in data:
            raise ValueError("Homography npz must contain key 'H'")
        H = data['H']
    else:
        vals = np.loadtxt(h_path).astype(np.float64).reshape(3, 3)
        H = vals
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H


def world_to_pixel(points_xy: np.ndarray,
                   H: Optional[np.ndarray] = None,
                   scale: Tuple[float, float] = (1.0, 1.0),
                   offset: Tuple[float, float] = (0.0, 0.0),
                   flip_y: bool = False,
                   image_height: Optional[int] = None) -> np.ndarray:
    """Convert scene/world coordinates (x, y) to image pixel positions.

    - If H is provided, applies homography transform.
    - Otherwise, applies affine-like scale+offset mapping.

    Args:
        points_xy: array of shape [..., 2] in world coordinates.
        H: 3x3 homography matrix (if available).
        scale: (sx, sy) scale factors for x and y.
        offset: (ox, oy) pixel offsets added after scaling.
        flip_y: if True, flips y-axis to account for different coordinate
                conventions (common in some datasets).
        image_height: needed only when flip_y=True and you want y=0 at bottom.
    Returns:
        Array of shape [..., 2] in pixel coordinates (float).
    """
    pts = points_xy.reshape(-1, 2).astype(np.float64)

    if H is not None:
        # Homogeneous coordinates
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        hom = np.concatenate([pts, ones], axis=1)  # [N,3]
        proj = (H @ hom.T).T                          # [N,3]
        uv = proj[:, :2] / proj[:, 2:3]
    else:
        sx, sy = scale
        ox, oy = offset
        uv = np.empty_like(pts)
        uv[:, 0] = pts[:, 0] * sx + ox
        uv[:, 1] = pts[:, 1] * sy + oy
        if flip_y:
            if image_height is None:
                raise ValueError("image_height required when flip_y=True")
            uv[:, 1] = image_height - uv[:, 1]

    return uv.reshape(points_xy.shape)


# ----------------------------
# Drawing helpers (OpenCV)
# ----------------------------

def draw_polyline(frame: np.ndarray, pts_px: np.ndarray, color: Tuple[int, int, int],
                  thickness: int = 2, alpha: float = 1.0) -> None:
    """Draw a polyline (sequence of points) with optional transparency."""
    if len(pts_px) < 2:
        return
    overlay = frame.copy()
    for i in range(len(pts_px) - 1):
        p1 = tuple(np.round(pts_px[i]).astype(int))
        p2 = tuple(np.round(pts_px[i + 1]).astype(int))
        cv2.line(overlay, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    else:
        frame[:] = overlay


def draw_point(frame: np.ndarray, p_px: np.ndarray, color: Tuple[int, int, int],
               radius: int = 4, thickness: int = -1, border_color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw a point with a white border for visibility."""
    center = tuple(np.round(p_px).astype(int))
    cv2.circle(frame, center, radius + 2, border_color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, thickness=thickness, lineType=cv2.LINE_AA)


def color_for_agent(agent_idx: int) -> Tuple[int, int, int]:
    """Deterministic BGR color for a given agent index (stable across frames)."""
    # Simple hashing into a palette; BGR for OpenCV
    palette = [
        (40, 180, 240), (60, 220, 60), (240, 180, 40), (200, 60, 200),
        (80, 160, 240), (240, 80, 120), (120, 220, 160), (200, 200, 60),
    ]
    return palette[agent_idx % len(palette)]


# ----------------------------
# Core overlay logic
# ----------------------------

def overlay_sequence_on_video(
    video_path: str,
    output_path: str,
    X_abs: torch.Tensor,           # [T_obs, N, 2] absolute observed positions
    Y_abs: torch.Tensor,           # [T_pred, N, 2] absolute future predictions
    mode: str = 'snapshot',
    H: Optional[np.ndarray] = None,
    scale: Tuple[float, float] = (1.0, 1.0),
    offset: Tuple[float, float] = (0.0, 0.0),
    flip_y: bool = False,
    start_frame: int = 0,
    fps_override: Optional[float] = None,
    snapshot_hold_seconds: float = 3.0,
) -> None:
    """Draw observed + predicted trajectories onto frames from a source video.

    Assumes `X_abs` corresponds to frames ending at `start_frame`. In temporal
    mode, predicted steps will be drawn on subsequent frames. If the source
    video has higher FPS than the dataset, you may want to pass an
    `fps_override` to control the output FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Helper to map a set of points (T, N, 2) to pixels
    def map_pts(tn2: np.ndarray) -> np.ndarray:
        return world_to_pixel(
            tn2, H=H, scale=scale, offset=offset, flip_y=flip_y, image_height=height
        )

    # Convert torch tensors to numpy
    X_abs_np = X_abs.detach().cpu().numpy()
    Y_abs_np = Y_abs.detach().cpu().numpy()

    # Seek to the desired start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - X_abs_np.shape[0] + 1))

    # 1) Write observed frames with observed trails
    for t in range(X_abs_np.shape[0]):
        ok, frame = cap.read()
        if not ok:
            break

        # Draw past trail up to current t
        obs_trail = X_abs_np[: t + 1]  # [t+1, N, 2]
        obs_trail_px = map_pts(obs_trail)
        for n in range(obs_trail_px.shape[1]):
            color = color_for_agent(n)
            draw_polyline(frame, obs_trail_px[:, n, :], color, thickness=2, alpha=0.8)
            draw_point(frame, obs_trail_px[-1, n, :], color)

        writer.write(frame)

    # 2) Snapshot or temporal rendering for predictions
    if mode == 'snapshot':
        # Draw the entire future on the last observed frame and hold for a few seconds
        ok, last_frame = cap.read()
        if not ok:
            # fallback: reuse previous frame if video ended
            last_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Compose a clean frame with last observed position drawn
        last_obs_px = map_pts(X_abs_np[-1:])[-1]
        for n in range(last_obs_px.shape[0]):
            color = color_for_agent(n)
            draw_point(last_frame, last_obs_px[n], color)

        # Draw full predicted polylines starting at last obs
        full_pred = np.concatenate([X_abs_np[-1:, :, :], X_abs_np[-1:, :, :] + (Y_abs_np.cumsum(axis=0))], axis=0)
        full_pred_px = map_pts(full_pred)
        for n in range(full_pred_px.shape[1]):
            color = color_for_agent(n)
            draw_polyline(last_frame, full_pred_px[:, n, :], color, thickness=2, alpha=0.9)

        hold_frames = int(fps * snapshot_hold_seconds)
        for _ in range(hold_frames):
            writer.write(last_frame)

    elif mode == 'temporal':
        # For each future step k, draw predicted positions at k
        # Y_abs is absolute; however, our model returns delta by default and
        # predict_trajectories already converted to absolute positions.
        for k in range(Y_abs_np.shape[0]):
            ok, frame = cap.read()
            if not ok:
                break

            # Draw entire observed trail for context
            obs_trail_px = map_pts(X_abs_np)
            for n in range(obs_trail_px.shape[1]):
                color = color_for_agent(n)
                draw_polyline(frame, obs_trail_px[:, n, :], color, thickness=2, alpha=0.4)
                draw_point(frame, obs_trail_px[-1, n, :], color)

            # Draw predicted path up to k for a smooth unfolding effect
            pred_prefix = np.concatenate([X_abs_np[-1:, :, :], Y_abs_np[: k + 1, :, :]], axis=0)
            pred_prefix_px = map_pts(pred_prefix)
            for n in range(pred_prefix_px.shape[1]):
                color = color_for_agent(n)
                draw_polyline(frame, pred_prefix_px[:, n, :], color, thickness=2, alpha=0.9)
                draw_point(frame, pred_prefix_px[-1, n, :], color)

            writer.write(frame)

    writer.release()
    cap.release()
    print(f"✅ Wrote overlay video to: {output_path}")


# ----------------------------
# Minimal demo runner
# ----------------------------

def run_demo(args: argparse.Namespace) -> None:
    """Load one batch, run the model, and overlay predictions to a video."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 1) Build model (load weights if you have a checkpoint).
    model = DMRGCN_GPGraph_Model(T_pred=args.t_pred)
    model.eval().to(device)

    # Optional: load a trained checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided; using randomly initialized model (for demo)")

    # 2) Build a minimal dataloader using TrajectoryDataset.
    # You may change data_dir to your desired scene split.
    data_dir = args.data_dir
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        obs_len=args.t_obs,
        pred_len=args.t_pred,
        skip=1,
        use_cache=True,
        cache_dir='./data_cache'
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    # Unpack tensors. Update these names to match your dataloader exactly.
    # Below we expect:
    #   obs_traj_abs: [T_obs, N, 2] absolute positions
    #   A_obs: [B, T_obs, N, N] or [B, N, N]
    #   M_obs: [B, T_obs, N]
    # If your loader uses different names or formats, adapt here.
    (obs_traj_abs, pred_traj_abs, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
     seq_start_end, agent_ids) = batch

    # Move tensors to device and to the shapes expected by the model.
    # The unified model expects X as [B, T_obs, N, d_in] with d_in>=2 (x,y in the first two dims).
    B = 1
    # obs_traj_abs: [T_obs, N, 2]
    X_abs = obs_traj_abs.unsqueeze(0).to(device).float()        # [1, T_obs, N, 2]
    # A_obs from collate_fn is shape [B, R, T, N, N]. Use distance channel (1) or aggregate.
    if A_obs.dim() == 5:
        A_obs = A_obs[:, 1, :, :, :]  # choose distance channel [B, T, N, N]
    A_obs = A_obs.to(device).float()
    M_obs = torch.ones(X_abs.shape[0], X_abs.shape[1], X_abs.shape[2], device=device)
    M_pred = torch.ones(B, args.t_pred, X_abs.shape[2], device=device)

    # 3) Run prediction in absolute coordinates
    with torch.no_grad():
        Y_abs = model.predict_trajectories(
            X_abs, A_obs, M_obs, M_pred, return_absolute=True
        )  # [B, T_pred, N, 2]

    # Squeeze batch dimension for overlay utilities
    X_abs_s = X_abs.squeeze(0).cpu()
    Y_abs_s = Y_abs.squeeze(0).cpu()

    # 4) Prepare mapping options
    H = load_homography(args.homography) if args.homography else None

    # 5) Render overlay
    overlay_sequence_on_video(
        video_path=args.video,
        output_path=args.output,
        X_abs=X_abs_s,               # [T_obs, N, 2]
        Y_abs=Y_abs_s,               # [T_pred, N, 2]
        mode=args.mode,
        H=H,
        scale=(args.scale_x, args.scale_y),
        offset=(args.offset_x, args.offset_y),
        flip_y=args.flip_y,
        start_frame=args.start_frame,
        fps_override=args.fps,
        snapshot_hold_seconds=args.snapshot_hold,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay DMRGCN+GPGraph predictions onto a video")

    # IO
    p.add_argument('--video', type=str, required=True, help='Path to input scene video')
    p.add_argument('--output', type=str, default='output/overlay_demo.mp4', help='Path to save MP4')
    p.add_argument('--checkpoint', type=str, default='', help='Path to trained model checkpoint')

    # Overlay behavior
    p.add_argument('--mode', type=str, default='snapshot', choices=['snapshot', 'temporal'])
    p.add_argument('--t-pred', type=int, default=12, help='Number of future steps to render')
    p.add_argument('--start-frame', type=int, default=0, help='Video frame index aligned to last obs')
    p.add_argument('--fps', type=float, default=None, help='Override output FPS (defaults to source FPS)')
    p.add_argument('--snapshot-hold', type=float, default=3.0, help='Seconds to hold snapshot frame')

    # Mapping options: either homography OR scale/offset
    p.add_argument('--homography', type=str, default='', help='npz/txt file containing 3x3 matrix H')
    p.add_argument('--scale', nargs=2, type=float, default=[1.0, 1.0], metavar=('SX', 'SY'), help='Scale factors')
    p.add_argument('--offset', nargs=2, type=float, default=[0.0, 0.0], metavar=('OX', 'OY'), help='Pixel offsets')
    p.add_argument('--flip-y', action='store_true', help='Flip y-axis after mapping (requires image height)')

    # Convenience flags
    p.add_argument('--cpu', action='store_true', help='Force CPU')
    p.add_argument('--data-dir', type=str, default='copy_dmrgcn/datasets/zara2/test', help='Directory of scene txt files')
    p.add_argument('--t-obs', type=int, default=8, help='Number of observed steps')

    args = p.parse_args()
    args.scale_x, args.scale_y = args.scale
    args.offset_x, args.offset_y = args.offset
    if not args.homography:
        args.homography = ''
    return args


if __name__ == '__main__':
    run_demo(parse_args())


