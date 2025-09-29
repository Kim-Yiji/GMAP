"""
Advanced Cache Management System for TrajectoryDataset
Provides robust caching with version control, validation, and error handling
"""

import os
import pickle
import hashlib
import json
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class CacheManager:
    """Advanced cache manager with version control and validation"""
    
    # Cache schema version - increment when data structure changes
    CACHE_VERSION = "2.0"
    
    # Required fields for cache validation
    REQUIRED_FIELDS = {
        'version': str,
        'created_at': str,
        'data_params': dict,
        'num_seq': int,
        'seq_list': np.ndarray,
        'seq_list_rel': np.ndarray,
        'agent_ids_list': np.ndarray,
        'loss_mask_list': np.ndarray,
        'non_linear_ped': (np.ndarray, torch.Tensor),
        'max_peds_in_frame': int,
        'num_peds_in_seq': list,
        'V_obs': list,
        'A_obs': list,
        'V_pred': list,
        'A_pred': list,
        'agent_ids_per_seq': list,
        'seq_start_end': list
    }
    
    def __init__(self, cache_dir: str = './data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def generate_cache_key(self, data_dir: str, params: Dict[str, Any]) -> str:
        """Generate unique cache key based on data and parameters"""
        # Get data directory name and file info
        data_dir_name = os.path.basename(os.path.normpath(data_dir))
        
        # Include file modification times
        file_info = []
        if os.path.exists(data_dir):
            for fname in sorted(os.listdir(data_dir)):
                fpath = os.path.join(data_dir, fname)
                if os.path.isfile(fpath):
                    mtime = os.path.getmtime(fpath)
                    file_info.append((fname, mtime))
        
        # Create hash from parameters, version, and file info
        params_str = f"{data_dir_name}_{params['obs_len']}_{params['pred_len']}_{params['skip']}_{params['delim']}"
        version_str = f"v{self.CACHE_VERSION}"
        files_str = str(file_info)
        combined_str = f"{version_str}_{params_str}_{files_str}"
        
        return hashlib.md5(combined_str.encode()).hexdigest()[:16]
    
    def get_cache_path(self, cache_key: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def is_cache_valid(self, cache_path: str, data_dir: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if cache is valid and compatible"""
        if not os.path.exists(cache_path):
            return False, "Cache file does not exist"
        
        try:
            # Load cache metadata first
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check version compatibility
            if 'version' not in cached_data:
                return False, "Cache missing version information"
            
            if cached_data['version'] != self.CACHE_VERSION:
                return False, f"Cache version mismatch: {cached_data['version']} vs {self.CACHE_VERSION}"
            
            # Check required fields
            for field, expected_type in self.REQUIRED_FIELDS.items():
                if field not in cached_data:
                    return False, f"Cache missing required field: {field}"
                
                if field == 'non_linear_ped':
                    # Special handling for non_linear_ped (can be numpy or tensor)
                    if not isinstance(cached_data[field], expected_type):
                        return False, f"Cache field {field} has wrong type"
                elif not isinstance(cached_data[field], expected_type):
                    return False, f"Cache field {field} has wrong type"
            
            # Check data parameters match
            if 'data_params' not in cached_data:
                return False, "Cache missing data parameters"
            
            cached_params = cached_data['data_params']
            for key in ['obs_len', 'pred_len', 'skip', 'delim']:
                if cached_params.get(key) != params.get(key):
                    return False, f"Parameter mismatch for {key}: {cached_params.get(key)} vs {params.get(key)}"
            
            return True, "Cache is valid"
            
        except Exception as e:
            return False, f"Cache validation error: {str(e)}"
    
    def save_cache(self, cache_path: str, data: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Save data to cache with metadata"""
        try:
            # Prepare cache data with metadata
            cache_data = {
                'version': self.CACHE_VERSION,
                'created_at': datetime.now().isoformat(),
                'data_params': params,
                **data
            }
            
            # Save to temporary file first, then rename (atomic operation)
            temp_path = cache_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            os.rename(temp_path, cache_path)
            print(f"💾 Cache saved successfully: {cache_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save cache: {str(e)}")
            # Clean up temp file if it exists
            temp_path = cache_path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def load_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """Load data from cache"""
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Remove metadata fields
            data = {k: v for k, v in cached_data.items() 
                   if k not in ['version', 'created_at', 'data_params']}
            
            print(f"✅ Cache loaded successfully: {cache_path}")
            print(f"   Created: {cached_data.get('created_at', 'Unknown')}")
            print(f"   Version: {cached_data.get('version', 'Unknown')}")
            return data
            
        except Exception as e:
            print(f"❌ Failed to load cache: {str(e)}")
            return None
    
    def normalize_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data types for consistency"""
        normalized = data.copy()
        
        # Ensure non_linear_ped is numpy array
        if 'non_linear_ped' in normalized:
            if isinstance(normalized['non_linear_ped'], torch.Tensor):
                normalized['non_linear_ped'] = normalized['non_linear_ped'].numpy()
        
        # Ensure num_peds_in_seq is list
        if 'num_peds_in_seq' in normalized:
            if isinstance(normalized['num_peds_in_seq'], np.ndarray):
                normalized['num_peds_in_seq'] = normalized['num_peds_in_seq'].tolist()
        
        return normalized
    
    def convert_to_tensors(self, data: Dict[str, Any], obs_len: int, pred_len: int) -> Dict[str, Any]:
        """Convert numpy arrays to torch tensors"""
        converted = data.copy()
        
        # Convert trajectory data
        if 'seq_list' in converted:
            converted['obs_traj'] = torch.from_numpy(converted['seq_list'][:, :, :obs_len]).type(torch.float)
            converted['pred_traj'] = torch.from_numpy(converted['seq_list'][:, :, obs_len:]).type(torch.float)
            converted['obs_traj_rel'] = torch.from_numpy(converted['seq_list_rel'][:, :, :obs_len]).type(torch.float)
            converted['pred_traj_rel'] = torch.from_numpy(converted['seq_list_rel'][:, :, obs_len:]).type(torch.float)
        
        if 'loss_mask_list' in converted:
            converted['loss_mask'] = torch.from_numpy(converted['loss_mask_list']).type(torch.float)
        
        if 'non_linear_ped' in converted:
            converted['non_linear_ped'] = torch.from_numpy(converted['non_linear_ped']).type(torch.float)
        
        if 'agent_ids_list' in converted:
            converted['agent_ids'] = torch.from_numpy(converted['agent_ids_list']).type(torch.long)
        
        # Reconstruct seq_start_end
        if 'num_peds_in_seq' in converted:
            cum_start_idx = [0] + np.cumsum(converted['num_peds_in_seq']).tolist()
            converted['seq_start_end'] = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        
        return converted
    
    def clear_cache(self, cache_key: str) -> bool:
        """Clear specific cache file"""
        cache_path = self.get_cache_path(cache_key)
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"🗑️  Cache cleared: {cache_path}")
                return True
        except Exception as e:
            print(f"❌ Failed to clear cache: {str(e)}")
        return False
    
    def clear_all_cache(self) -> bool:
        """Clear all cache files"""
        try:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, fname))
            print(f"🗑️  All cache files cleared from {self.cache_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to clear all cache: {str(e)}")
        return False
