"""
Homography parser for Stanford Drone Dataset (SDD).

This module parses the H_SDD.txt file which contains calibration data
for converting world coordinates to pixel coordinates.
"""

import os
import re
from typing import Dict, Optional, Tuple
import numpy as np


def parse_homography_file(homography_path: str) -> Dict[str, Dict]:
    """
    Parse the H_SDD.txt homography file.
    
    Args:
        homography_path: Path to H_SDD.txt file
    
    Returns:
        Dictionary mapping scene names to calibration data:
        {
            'bookstore_0': {
                'dataset': 'Bookstore',
                'pixel_x': [349, 350],
                'pixel_y': [15, 16],
                'pixel_dist': [349.32, 350.37],
                'meters': 13.4112,
                'ratio': 0.0383,  # pixels per meter (average)
                'reference_point': (349.5, 15.5)  # Average pixel coordinates
            },
            ...
        }
    """
    if not os.path.exists(homography_path):
        raise FileNotFoundError(f"Homography file not found: {homography_path}")
    
    calibration_data = {}
    
    with open(homography_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Parse line: File, Dataset, Nr, Version, Feet, Inch, Meters, Pixel X, Pixel Y, Pixel Dist, Average, Diff, Ratio
        parts = line.split('\t')
        if len(parts) < 12:
            continue
        
        filename = parts[0].strip()
        dataset = parts[1].strip()
        nr = parts[2].strip()
        version = parts[3].strip()
        
        try:
            meters = float(parts[6])
            pixel_x = float(parts[7])
            pixel_y = float(parts[8])
            pixel_dist = float(parts[9])
            ratio = float(parts[11])
        except (ValueError, IndexError):
            continue
        
        # Create scene key (e.g., 'bookstore_0' from 'bookstore_0.jpg')
        scene_key = filename.replace('.jpg', '')
        
        if scene_key not in calibration_data:
            calibration_data[scene_key] = {
                'dataset': dataset,
                'nr': nr,
                'pixel_x': [],
                'pixel_y': [],
                'pixel_dist': [],
                'meters': meters,
                'ratio': [],
                'reference_point_a': None,
                'reference_point_b': None,
            }
        
        # Store measurements for both versions (A and B)
        calibration_data[scene_key]['pixel_x'].append(pixel_x)
        calibration_data[scene_key]['pixel_y'].append(pixel_y)
        calibration_data[scene_key]['pixel_dist'].append(pixel_dist)
        calibration_data[scene_key]['ratio'].append(ratio)
        
        # Store reference point (first measurement of each version)
        if version == 'A' and calibration_data[scene_key]['reference_point_a'] is None:
            calibration_data[scene_key]['reference_point_a'] = (pixel_x, pixel_y)
        elif version == 'B' and calibration_data[scene_key]['reference_point_b'] is None:
            calibration_data[scene_key]['reference_point_b'] = (pixel_x, pixel_y)
    
    # Calculate averages for each scene
    for scene_key in calibration_data:
        data = calibration_data[scene_key]
        
        # The ratio column is meters per pixel (small number like 0.038)
        # Store it for reference, but we'll calculate pixels_per_meter from pixel_dist / meters
        if data['ratio']:
            data['ratio_avg'] = np.mean(data['ratio'])  # This is meters per pixel
        else:
            data['ratio_avg'] = None
        
        # Calculate pixels per meter from actual measurements
        if data['pixel_dist'] and data['meters']:
            data['pixels_per_meter'] = np.mean(data['pixel_dist']) / data['meters']
        elif data['ratio_avg'] and data['ratio_avg'] > 0:
            # Fallback: use inverse of ratio
            data['pixels_per_meter'] = 1.0 / data['ratio_avg']
        else:
            data['pixels_per_meter'] = None
        
        # Average reference point
        if data['reference_point_a'] and data['reference_point_b']:
            ref_a = data['reference_point_a']
            ref_b = data['reference_point_b']
            data['reference_point'] = (
                (ref_a[0] + ref_b[0]) / 2,
                (ref_a[1] + ref_b[1]) / 2
            )
        elif data['reference_point_a']:
            data['reference_point'] = data['reference_point_a']
        elif data['reference_point_b']:
            data['reference_point'] = data['reference_point_b']
        else:
            # Use average of all pixel coordinates
            if data['pixel_x'] and data['pixel_y']:
                data['reference_point'] = (
                    np.mean(data['pixel_x']),
                    np.mean(data['pixel_y'])
                )
            else:
                data['reference_point'] = (0, 0)
        
        # Clean up intermediate data
        del data['reference_point_a']
        del data['reference_point_b']
    
    return calibration_data


def get_scene_calibration(homography_path: str, scene_name: str, video_number: int = 0) -> Optional[Dict]:
    """
    Get calibration data for a specific scene.
    
    Args:
        homography_path: Path to H_SDD.txt file
        scene_name: Scene name (e.g., 'bookstore', 'deathCircle', 'gates')
        video_number: Video number (default: 0)
    
    Returns:
        Calibration data dictionary or None if not found
    """
    calibration_data = parse_homography_file(homography_path)
    
    # Try to find the scene (case-insensitive)
    scene_key = f"{scene_name.lower()}_{video_number}"
    
    # Try exact match first
    if scene_key in calibration_data:
        return calibration_data[scene_key]
    
    # Try case-insensitive search
    for key in calibration_data.keys():
        if key.lower() == scene_key.lower():
            return calibration_data[key]
    
    # Try matching dataset name
    for key, data in calibration_data.items():
        if data['dataset'].lower() == scene_name.lower() and data['nr'] == str(video_number):
            return data
    
    return None


def calculate_transform_from_homography(
    homography_path: str,
    scene_name: str,
    video_number: int,
    video_width: int,
    video_height: int,
    annotation_min_x: float,
    annotation_max_x: float,
    annotation_min_y: float,
    annotation_max_y: float
) -> Tuple[float, float, float, float]:
    """
    Calculate coordinate transformation parameters using homography data.
    
    Args:
        homography_path: Path to H_SDD.txt file
        scene_name: Scene name (e.g., 'bookstore')
        video_number: Video number (default: 0)
        video_width: Video width in pixels
        video_height: Video height in pixels
        annotation_min_x: Minimum X coordinate in annotations
        annotation_max_x: Maximum X coordinate in annotations
        annotation_min_y: Minimum Y coordinate in annotations
        annotation_max_y: Maximum Y coordinate in annotations
    
    Returns:
        Tuple of (scale_x, scale_y, offset_x, offset_y)
    """
    calibration = get_scene_calibration(homography_path, scene_name, video_number)
    
    if calibration is None:
        raise ValueError(
            f"Could not find calibration data for scene '{scene_name}' video {video_number}. "
            f"Available scenes: {list(parse_homography_file(homography_path).keys())}"
        )
    
    # Get pixels per meter from calibration data
    pixels_per_meter = calibration.get('pixels_per_meter')
    if pixels_per_meter is None:
        raise ValueError(f"No valid pixels per meter data found for scene '{scene_name}' video {video_number}")
    
    # The annotations are in world coordinates (meters)
    # Scale factor: pixels per meter from homography
    scale_x = pixels_per_meter
    scale_y = pixels_per_meter
    
    # Get reference point (this is a known point in pixel coordinates)
    ref_x, ref_y = calibration['reference_point']
    
    # The reference point corresponds to where the measurement was taken
    # However, we don't know the exact annotation coordinates for this reference point
    # So we need to use a different strategy:
    # 
    # Option 1: Assume annotations start near (0,0) and map to top-left of video
    # Option 2: Use the reference point as a known correspondence and calculate offset
    # Option 3: Scale annotations and adjust offset to fit within video bounds
    
    # For now, let's try mapping annotation min to a margin from top-left
    # and see if that aligns better. We can also try using the reference point.
    
    # Calculate where annotations would be if we just scale them
    scaled_min_x = annotation_min_x * scale_x
    scaled_max_x = annotation_max_x * scale_x
    scaled_min_y = annotation_min_y * scale_y
    scaled_max_y = annotation_max_y * scale_y
    
    # Calculate offset to position annotations in video
    # Try centering or aligning to reference point
    margin = 50  # Margin from edges
    
    # Option: Map annotation min to margin
    offset_x = margin - scaled_min_x
    offset_y = margin - scaled_min_y
    
    # Alternative: Try to use reference point if we can infer annotation coordinates
    # For now, use the margin approach but allow override via manual offset
    
    return scale_x, scale_y, offset_x, offset_y

