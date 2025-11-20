"""
Threat Score Visualization for DMRGCN
특정 인물(track_id)에 대한 threat score를 영상에 시각화
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List, Dict
import matplotlib.patches as mpatches


def get_threat_color(threat_score: float, colormap: str = 'RdYlGn_r') -> Tuple[int, int, int]:
    """Convert threat score (0-1) to BGR color for OpenCV.
    
    Args:
        threat_score: Threat score in [0, 1], where 1 is high threat
        colormap: Matplotlib colormap name (default: 'RdYlGn_r' for red=high threat)
    
    Returns:
        BGR color tuple (B, G, R) in [0, 255]
    """
    # Clamp score to [0, 1]
    score = np.clip(threat_score, 0.0, 1.0)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    rgba = cmap(score)  # Returns (R, G, B, A) in [0, 1]
    
    # Convert to BGR and scale to [0, 255]
    bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
    return bgr


def draw_threat_arrow(frame: np.ndarray, 
                     pos1: Tuple[float, float],
                     pos2: Tuple[float, float],
                     threat_score: float,
                     thickness: int = 2,
                     min_threat: float = 0.1) -> np.ndarray:
    """Draw an arrow between two positions with color based on threat score.
    
    Args:
        frame: Image frame (H, W, 3) in BGR format
        pos1: Start position (x, y) in pixel coordinates
        pos2: End position (x, y) in pixel coordinates
        threat_score: Threat score in [0, 1]
        thickness: Arrow thickness
        min_threat: Minimum threat score to draw (below this, skip)
    
    Returns:
        Frame with arrow drawn
    """
    if threat_score < min_threat:
        return frame
    
    # Get color based on threat score
    color = get_threat_color(threat_score, colormap='RdYlGn_r')
    
    # Draw arrow
    pt1 = (int(pos1[0]), int(pos1[1]))
    pt2 = (int(pos2[0]), int(pos2[1]))
    
    # Draw line with thickness proportional to threat
    line_thickness = max(1, int(thickness * (0.5 + 0.5 * threat_score)))
    cv2.arrowedLine(frame, pt1, pt2, color, line_thickness, 
                   tipLength=0.3, line_type=cv2.LINE_AA)
    
    return frame


def draw_threat_circle(frame: np.ndarray,
                      center: Tuple[float, float],
                      threat_score: float,
                      radius: int = 15,
                      min_threat: float = 0.1) -> np.ndarray:
    """Draw a circle at position with color and size based on threat score.
    
    Args:
        frame: Image frame (H, W, 3) in BGR format
        center: Center position (x, y) in pixel coordinates
        threat_score: Threat score in [0, 1]
        radius: Base radius
        min_threat: Minimum threat score to draw
    
    Returns:
        Frame with circle drawn
    """
    if threat_score < min_threat:
        return frame
    
    # Get color based on threat score
    color = get_threat_color(threat_score, colormap='RdYlGn_r')
    
    # Adjust radius based on threat (higher threat = larger circle)
    adjusted_radius = int(radius * (0.7 + 0.3 * threat_score))
    
    # Draw filled circle
    center_int = (int(center[0]), int(center[1]))
    cv2.circle(frame, center_int, adjusted_radius, color, -1)
    
    # Draw border
    cv2.circle(frame, center_int, adjusted_radius, (255, 255, 255), 2)
    
    return frame


def draw_threat_heatmap_overlay(frame: np.ndarray,
                                positions: np.ndarray,
                                threat_matrix: np.ndarray,
                                target_idx: int,
                                alpha: float = 0.3,
                                kernel_size: int = 50) -> np.ndarray:
    """Draw a heatmap overlay showing threat around target person.
    
    Args:
        frame: Image frame (H, W, 3) in BGR format
        positions: (N, 2) array of positions in pixel coordinates
        threat_matrix: (N, N) threat score matrix
        target_idx: Index of target person
        alpha: Transparency of heatmap overlay
        kernel_size: Size of Gaussian kernel for smoothing
    
    Returns:
        Frame with heatmap overlay
    """
    h, w = frame.shape[:2]
    
    # Get threat scores for target person (row)
    threat_scores = threat_matrix[target_idx, :]  # (N,)
    
    # Create heatmap image
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # For each person, add Gaussian blob at their position
    for i, (x, y) in enumerate(positions):
        if i == target_idx:
            continue  # Skip target itself
        
        score = threat_scores[i]
        if score < 0.1:
            continue
        
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < w and 0 <= y_int < h:
            # Create Gaussian kernel
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_sq = (x_coords - x_int) ** 2 + (y_coords - y_int) ** 2
            gaussian = np.exp(-dist_sq / (2 * (kernel_size / 3) ** 2))
            
            # Add to heatmap with threat score as weight
            heatmap += gaussian * score
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Convert to color
    heatmap_colored = plt.get_cmap('RdYlGn_r')(heatmap)[:, :, :3]  # (H, W, 3) RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    
    # Blend with original frame
    frame_blended = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
    
    return frame_blended


def visualize_threat_scores_for_person(
    obs_traj: torch.Tensor,
    A_obs: torch.Tensor,
    target_track_id: int,
    track_ids: List[int],
    frame_idx: int,
    video_frame: Optional[np.ndarray] = None,
    scale: float = 1.0,
    offset: Tuple[float, float] = (0, 0),
    visualize_mode: str = 'arrows',  # 'arrows', 'circles', 'heatmap', 'all'
    min_threat: float = 0.1,
    draw_trajectory: bool = True,
) -> np.ndarray:
    """Visualize threat scores for a specific person at a given frame.
    
    Args:
        obs_traj: (N, 2, T) absolute trajectory tensor
        A_obs: (4, T, N, N) adjacency tensor [disp, dist, pp_threat, po_threat]
        target_track_id: Track ID of target person
        track_ids: List of track IDs corresponding to obs_traj indices
        frame_idx: Frame index within observation window (0 to obs_len-1)
        video_frame: Optional video frame to overlay on (H, W, 3) BGR
        scale: Scale factor for coordinates (e.g., 1000.0 for normalized SDD)
        offset: (x, y) offset for coordinate translation
        visualize_mode: Visualization mode
        min_threat: Minimum threat score to visualize
        draw_trajectory: Whether to draw trajectory history
    
    Returns:
        Visualized frame (H, W, 3) BGR
    """
    # Find target index
    try:
        target_idx = track_ids.index(target_track_id)
    except ValueError:
        raise ValueError(f"Target track_id {target_track_id} not found in track_ids")
    
    # Get frame dimensions
    if video_frame is not None:
        frame = video_frame.copy()
        h, w = frame.shape[:2]
    else:
        # Create black background
        # Estimate frame size from trajectory bounds
        all_pos = obs_traj[:, :, :frame_idx + 1].cpu().numpy()
        all_pos = all_pos * scale + np.array(offset)
        min_x, max_x = all_pos[:, 0, :].min(), all_pos[:, 0, :].max()
        min_y, max_y = all_pos[:, 1, :].min(), all_pos[:, 1, :].max()
        
        # Add padding
        padding = 50
        w = int(max_x - min_x + 2 * padding)
        h = int(max_y - min_y + 2 * padding)
        offset = (offset[0] - min_x + padding, offset[1] - min_y + padding)
        
        frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get positions at current frame
    positions = obs_traj[:, :, frame_idx].cpu().numpy()  # (N, 2)
    positions = positions * scale + np.array(offset)
    
    # Get threat matrices (PP-threat and PO-threat)
    # A_obs shape: (4, T, N, N) where 4 = [disp, dist, pp_threat, po_threat]
    pp_threat = A_obs[2, frame_idx].cpu().numpy()  # (N, N) PP-threat
    po_threat = A_obs[3, frame_idx].cpu().numpy()  # (N, N) PO-threat
    
    # Combine threats (PP for pedestrian pairs, PO for pedestrian-object pairs)
    # For visualization, we'll use pp_threat for all pairs and add po_threat where applicable
    threat_matrix = pp_threat.copy()
    threat_matrix = np.maximum(threat_matrix, po_threat)  # Take maximum of PP and PO threat
    
    # Draw trajectory history if requested
    if draw_trajectory:
        for i in range(obs_traj.shape[0]):
            traj = obs_traj[i, :, :frame_idx + 1].cpu().numpy()  # (2, T)
            traj = traj.T * scale + np.array(offset)  # (T, 2)
            
            color = (0, 255, 0) if i == target_idx else (100, 100, 100)
            thickness = 3 if i == target_idx else 1
            
            for t in range(len(traj) - 1):
                pt1 = (int(traj[t, 0]), int(traj[t, 1]))
                pt2 = (int(traj[t + 1, 0]), int(traj[t + 1, 1]))
                cv2.line(frame, pt1, pt2, color, thickness)
    
    # Get target position
    target_pos = positions[target_idx]
    
    # Visualize based on mode
    if visualize_mode in ['arrows', 'all']:
        # Draw arrows from target to others
        for i in range(len(positions)):
            if i == target_idx:
                continue
            
            threat = threat_matrix[target_idx, i]
            if threat >= min_threat:
                frame = draw_threat_arrow(frame, target_pos, positions[i], threat, thickness=3)
    
    if visualize_mode in ['circles', 'all']:
        # Draw circles at each person's position with threat-based color
        for i in range(len(positions)):
            if i == target_idx:
                # Highlight target with special color
                cv2.circle(frame, (int(positions[i, 0]), int(positions[i, 1])), 
                          20, (255, 255, 0), -1)  # Yellow for target
                cv2.circle(frame, (int(positions[i, 0]), int(positions[i, 1])), 
                          20, (0, 0, 0), 2)  # Black border
            else:
                threat = threat_matrix[target_idx, i]
                if threat >= min_threat:
                    frame = draw_threat_circle(frame, positions[i], threat, radius=12)
                else:
                    # Draw gray circle for low threat
                    cv2.circle(frame, (int(positions[i, 0]), int(positions[i, 1])), 
                              8, (128, 128, 128), -1)
    
    if visualize_mode in ['heatmap', 'all']:
        # Draw heatmap overlay
        frame = draw_threat_heatmap_overlay(frame, positions, threat_matrix, 
                                            target_idx, alpha=0.3)
    
    # Add text label for target
    cv2.putText(frame, f'Target ID: {target_track_id}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Frame: {frame_idx}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add threat score legend
    max_threat = threat_matrix[target_idx, :].max()
    if max_threat > 0:
        cv2.putText(frame, f'Max Threat: {max_threat:.2f}', (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def create_threat_visualization_video(
    dataset,
    model,
    target_track_id: int,
    sequence_idx: int = 0,
    output_path: str = './threat_visualization.mp4',
    video_path: Optional[str] = None,
    scale: float = 1000.0,
    visualize_mode: str = 'all',
    fps: float = 10.0,
    device: str = 'cuda:0'
):
    """Create a video visualization of threat scores for a specific person.
    
    Args:
        dataset: SDDTrajectoryDataset instance
        model: Trained DMRGCN model
        target_track_id: Track ID of target person to visualize
        sequence_idx: Index of sequence in dataset
        output_path: Path to save output video
        video_path: Optional path to original video file
        scale: Scale factor for coordinates
        visualize_mode: Visualization mode ('arrows', 'circles', 'heatmap', 'all')
        fps: Output video frame rate
        device: Device to run model on
    """
    # Get sequence data
    batch = dataset[sequence_idx]
    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = batch[:4]
    V_obs, A_obs = batch[6], batch[7]
    
    # Get track IDs from dataset (if available)
    if hasattr(dataset, 'get_track_ids'):
        track_ids = dataset.get_track_ids(sequence_idx)
    elif len(batch) > 10:  # If track_ids are in the batch
        track_ids = batch[10]
    else:
        # Fallback: use indices as track IDs
        num_tracks = obs_traj.shape[0]
        track_ids = list(range(num_tracks))
    
    # Move to device
    obs_traj = obs_traj.to(device)
    A_obs = A_obs.to(device)
    
    # Load video if provided
    cap = None
    video_width, video_height = 1920, 1080  # Default SDD resolution
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else fps
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    obs_len = V_obs.shape[0]
    
    # Process each frame
    for frame_idx in range(obs_len):
        # Get video frame if available
        video_frame = None
        if cap is not None:
            ret, video_frame = cap.read()
            if not ret:
                video_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        else:
            video_frame = None
        
        # Create visualization
        vis_frame = visualize_threat_scores_for_person(
            obs_traj=obs_traj,
            A_obs=A_obs,
            target_track_id=target_track_id,
            track_ids=track_ids,
            frame_idx=frame_idx,
            video_frame=video_frame,
            scale=scale,
            offset=(0, 0),
            visualize_mode=visualize_mode,
            min_threat=0.1,
            draw_trajectory=True
        )
        
        # Resize if needed
        if vis_frame.shape[:2] != (video_height, video_width):
            vis_frame = cv2.resize(vis_frame, (video_width, video_height))
        
        out.write(vis_frame)
    
    # Cleanup
    out.release()
    if cap is not None:
        cap.release()
    
    print(f"Threat visualization video saved to: {output_path}")


def save_threat_visualization_frames(
    dataset,
    model,
    target_track_id: int,
    sequence_idx: int = 0,
    output_dir: str = './threat_frames/',
    video_path: Optional[str] = None,
    scale: float = 1000.0,
    visualize_mode: str = 'all',
    device: str = 'cuda:0'
):
    """Save individual frames as images instead of video.
    
    Args are same as create_threat_visualization_video except output_path -> output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sequence data
    batch = dataset[sequence_idx]
    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = batch[:4]
    V_obs, A_obs = batch[6], batch[7]
    
    # Get track IDs from dataset (if available)
    if hasattr(dataset, 'get_track_ids'):
        track_ids = dataset.get_track_ids(sequence_idx)
    elif len(batch) > 10:  # If track_ids are in the batch
        track_ids = batch[10]
    else:
        # Fallback: use indices as track IDs
        num_tracks = obs_traj.shape[0]
        track_ids = list(range(num_tracks))
    
    # Move to device
    obs_traj = obs_traj.to(device)
    A_obs = A_obs.to(device)
    
    # Load video if provided
    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    
    obs_len = V_obs.shape[0]
    
    # Process each frame
    for frame_idx in range(obs_len):
        # Get video frame if available
        video_frame = None
        if cap is not None:
            ret, video_frame = cap.read()
            if not ret:
                video_frame = None
        
        # Create visualization
        vis_frame = visualize_threat_scores_for_person(
            obs_traj=obs_traj,
            A_obs=A_obs,
            target_track_id=target_track_id,
            track_ids=track_ids,
            frame_idx=frame_idx,
            video_frame=video_frame,
            scale=scale,
            offset=(0, 0),
            visualize_mode=visualize_mode,
            min_threat=0.1,
            draw_trajectory=True
        )
        
        # Save frame
        output_path = os.path.join(output_dir, f'frame_{frame_idx:03d}.jpg')
        cv2.imwrite(output_path, vis_frame)
    
    if cap is not None:
        cap.release()
    
    print(f"Threat visualization frames saved to: {output_dir}")

