import pandas as pd
import numpy as np
import os

def calculate_centroid(row):
    """Calculates the centroid of a bounding box."""
    return (row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2

def process_video_data_with_scoring(video_path, labels_path, target_pedestrian_id):
    """
    Processes annotation data to create a detailed risk scoring dataset for a target pedestrian.

    The risk score is based on:
    1. d_ij: Euclidean distance to an obstacle.
    2. v_ij: Relative approach speed towards the obstacle.
    3. size_j: The size of the obstacle based on its label.
    4. TTC_ij: Time-to-collision with the obstacle.

    Args:
        video_path (str): Path to the annotations.csv file.
        labels_path (str): Path to the _labels.csv file.
        target_pedestrian_id (int): The track_id of the target pedestrian.

    Returns:
        pd.DataFrame: A DataFrame with detailed risk scores for the target pedestrian
                      relative to other objects in each frame.
    """
    try:
        annotations_df = pd.read_csv(video_path)
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None

    # --- 1. Data Preparation ---
    # Drop the 'label' column from the main annotations file to use the one from the labels file as the ground truth.
    if 'label' in annotations_df.columns:
        annotations_df = annotations_df.drop(columns=['label'])
        
    annotations_df = pd.merge(annotations_df, labels_df, on='track_id', how='left')
    annotations_df.rename(columns={'label': 'object_label'}, inplace=True)
    annotations_df['centroid_x'], annotations_df['centroid_y'] = zip(*annotations_df.apply(calculate_centroid, axis=1))

    # Check if the target is a pedestrian
    target_info = labels_df[labels_df['track_id'] == target_pedestrian_id]
    if target_info.empty or target_info.iloc[0]['label'] != 'Pedestrian':
        print(f"Error: Target track_id {target_pedestrian_id} is not a Pedestrian or does not exist.")
        return None

    # --- 2. Velocity Calculation ---
    annotations_df.sort_values(by=['track_id', 'frame'], inplace=True)
    annotations_df['vx'] = annotations_df.groupby('track_id')['centroid_x'].diff()
    annotations_df['vy'] = annotations_df.groupby('track_id')['centroid_y'].diff()
    # Assuming frame rate is constant, velocity is in pixels/frame.
    annotations_df.fillna({'vx': 0, 'vy': 0}, inplace=True)

    # --- 3. Pairwise Score Calculation ---
    target_frames = annotations_df[annotations_df['track_id'] == target_pedestrian_id]['frame'].unique()
    all_pairs_data = []

    # Define size mapping based on social consensus
    label_sizes = {'Pedestrian': 1.0, 'Biker': 2.0, 'Skater': 2.0, 'Cart': 3.0, 'Car': 5.0, 'Bus': 8.0}

    for frame in target_frames:
        frame_df = annotations_df[annotations_df['frame'] == frame].set_index('track_id')
        target_series = frame_df.loc[target_pedestrian_id]
        
        other_objects_df = frame_df.drop(target_pedestrian_id)

        for other_id, other_series in other_objects_df.iterrows():
            # d_ij: Euclidean Distance
            pos_vector = np.array([other_series['centroid_x'] - target_series['centroid_x'],
                                   other_series['centroid_y'] - target_series['centroid_y']])
            distance = np.linalg.norm(pos_vector)
            if distance == 0: continue

            # v_ij: Relative Approach Speed
            rel_vel_vector = np.array([other_series['vx'] - target_series['vx'],
                                       other_series['vy'] - target_series['vy']])
            approach_speed = -np.dot(rel_vel_vector, pos_vector) / distance
            
            # size_j: Obstacle Size
            obstacle_size = label_sizes.get(other_series['object_label'], 1.0) # Default to 1.0 if label not in map

            # TTC_ij: Time-to-Collision
            ttc = distance / approach_speed if approach_speed > 1e-6 else np.inf # Avoid division by zero/small numbers

            all_pairs_data.append({
                'frame': frame,
                'target_id': target_pedestrian_id,
                'other_id': other_id,
                'other_label': other_series['object_label'],
                'distance': distance,
                'approach_speed': approach_speed,
                'obstacle_size': obstacle_size,
                'ttc': ttc
            })

    if not all_pairs_data:
        return pd.DataFrame()

    scores_df = pd.DataFrame(all_pairs_data)

    # --- 4. Normalization and Final Scoring ---
    # Only consider positive approach speeds for risk
    scores_df['approach_speed'] = scores_df['approach_speed'].clip(lower=0)
    
    # Use inverse TTC, handling infinity. Higher inverse_ttc means higher risk.
    scores_df['inverse_ttc'] = 1 / scores_df['ttc']
    scores_df.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf from 1/0 with 0

    # Normalize each component to [0, 1] where 1 is highest risk
    # Distance: lower is riskier
    max_dist = scores_df['distance'].max()
    scores_df['norm_distance'] = (1 - scores_df['distance'] / max_dist) if max_dist > 0 else 0

    # Approach Speed: higher is riskier
    max_speed = scores_df['approach_speed'].max()
    scores_df['norm_speed'] = scores_df['approach_speed'] / max_speed if max_speed > 0 else 0

    # Obstacle Size: larger is riskier
    max_size = scores_df['obstacle_size'].max()
    scores_df['norm_size'] = scores_df['obstacle_size'] / max_size if max_size > 0 else 0

    # Inverse TTC: higher is riskier
    max_inv_ttc = scores_df['inverse_ttc'].max()
    scores_df['norm_ttc'] = scores_df['inverse_ttc'] / max_inv_ttc if max_inv_ttc > 0 else 0

    # Final Risk Score (1:1:1:1 weighting)
    scores_df['risk_score'] = (scores_df['norm_distance'] + 
                               scores_df['norm_speed'] + 
                               scores_df['norm_size'] + 
                               scores_df['norm_ttc'])

    return scores_df

if __name__ == "__main__":
    base_dir = "/Users/yiji/Downloads/SDD_visualize/SDD_datasets"
    dataset_name = "coupa"
    video_id = "video0"
    target_pedestrian_id = 1 # Must be a pedestrian

    annotations_file = os.path.join(base_dir, "SDD_raw", dataset_name, video_id, "annotations.csv")
    labels_file = os.path.join(base_dir, "SDD_labels", dataset_name, f"{video_id}_labels.csv")

    print(f"Processing {dataset_name}/{video_id} for target pedestrian {target_pedestrian_id}...")
    
    final_scores_df = process_video_data_with_scoring(annotations_file, labels_file, target_pedestrian_id)

    if final_scores_df is not None and not final_scores_df.empty:
        print("\nFinal Scores DataFrame (first 5 rows):")
        print(final_scores_df.head())

        # Save to CSV
        output_dir = os.path.join(base_dir, "processed_for_visualization", dataset_name, video_id)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{video_id}_target_{target_pedestrian_id}_risk_scores.csv")
        final_scores_df.to_csv(output_path, index=False)
        
        print(f"\nProcessed risk scores saved to: {output_path}")
    else:
        print("Data processing failed or no data was generated.")