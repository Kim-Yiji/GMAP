"""
Data parser module for SDD annotation files.

This module loads annotation files in the format:
    frame_id    ped_id    x    y

And organizes the data by frame for easy access during threat score computation.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def load_annotations(annotation_path: str, delim: str = '\t') -> np.ndarray:
    """
    Load annotation file and return as numpy array.
    
    Args:
        annotation_path: Path to the annotation file
        delim: Delimiter used in the file (default: tab)
    
    Returns:
        numpy array with columns [frame_id, ped_id, x, y]
    """
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    
    with open(annotation_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim)
            try:
                parts = [float(i) for i in parts]
                if len(parts) >= 4:  # Ensure we have at least frame_id, ped_id, x, y
                    data.append(parts[:4])  # Take first 4 columns
            except ValueError:
                # Skip lines that cannot be parsed as floats
                continue
    
    if not data:
        return np.array([]).reshape(0, 4)
    
    return np.asarray(data)


def organize_by_frame(annotations: np.ndarray) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Organize annotations by frame_id.
    
    Args:
        annotations: numpy array with columns [frame_id, ped_id, x, y]
    
    Returns:
        Dictionary mapping frame_id to list of (ped_id, x, y) tuples
    """
    frame_data = defaultdict(list)
    
    for row in annotations:
        frame_id = int(row[0])
        ped_id = int(row[1])
        x = float(row[2])
        y = float(row[3])
        frame_data[frame_id].append((ped_id, x, y))
    
    # Convert defaultdict to regular dict and sort by frame_id
    return dict(sorted(frame_data.items()))


def get_object_positions_per_frame(annotation_path: str) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Convenience function to load annotations and organize by frame.
    
    Args:
        annotation_path: Path to the annotation file
    
    Returns:
        Dictionary mapping frame_id to list of (ped_id, x, y) tuples
    """
    annotations = load_annotations(annotation_path)
    return organize_by_frame(annotations)


def get_available_object_ids(annotations: np.ndarray) -> List[int]:
    """
    Get list of all unique object IDs in the annotations.
    
    Args:
        annotations: numpy array with columns [frame_id, ped_id, x, y]
    
    Returns:
        Sorted list of unique object IDs
    """
    if len(annotations) == 0:
        return []
    
    unique_ids = np.unique(annotations[:, 1].astype(int))
    return sorted(unique_ids.tolist())


def get_frame_range(annotations: np.ndarray) -> Tuple[int, int]:
    """
    Get the min and max frame IDs in the annotations.
    
    Args:
        annotations: numpy array with columns [frame_id, ped_id, x, y]
    
    Returns:
        Tuple of (min_frame_id, max_frame_id)
    """
    if len(annotations) == 0:
        return (0, 0)
    
    min_frame = int(np.min(annotations[:, 0]))
    max_frame = int(np.max(annotations[:, 0]))
    return (min_frame, max_frame)

