import os
import pickle
import torch
import numpy as np
from datetime import datetime


def save_group_data(result_dict, dataset_name, frame_id=None):
    """Save group assignment results to pickle file
    
    Args:
        result_dict: Dictionary containing group assignment results
        dataset_name: Name of the dataset (e.g., 'eth', 'hotel')
        frame_id: Optional frame identifier
        
    Returns:
        str: Path to the saved pickle file
    """
    # Create cache directory if it doesn't exist
    os.makedirs('data_cache', exist_ok=True)
    
    # Generate filename with timestamp and dataset name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    frame_str = f'_frame{frame_id}' if frame_id is not None else ''
    filename = f'group_data_{dataset_name}{frame_str}_{timestamp}.pkl'
    filepath = os.path.join('data_cache', filename)
    
    # Save data
    with open(filepath, 'wb') as f:
        pickle.dump(result_dict, f)
    
    return filepath


def load_group_data(filepath):
    """Load group assignment results from pickle file
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        dict: Dictionary containing group assignment results
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")
    
    with open(filepath, 'rb') as f:
        result_dict = pickle.load(f)
    
    return result_dict


def numpy_to_torch(result_dict):
    """Convert numpy arrays in result dict to PyTorch tensors
    
    Args:
        result_dict: Dictionary containing numpy arrays
        
    Returns:
        dict: Dictionary with PyTorch tensors
    """
    torch_dict = {}
    
    for key, value in result_dict.items():
        if isinstance(value, np.ndarray):
            torch_dict[key] = torch.from_numpy(value)
        else:
            torch_dict[key] = value
            
    return torch_dict