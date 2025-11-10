#!/usr/bin/env python3
"""Quick script to analyze the output metadata."""

import json
import sys

def analyze_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("Threat Score Visualization Results")
    print("=" * 60)
    print(f"\nTarget ID: {data['target_id']}")
    print(f"Total frames processed: {len(data['frames'])}")
    print(f"Frame range: {data['frames'][0]['frame_id']} to {data['frames'][-1]['frame_id']}")
    print(f"\nVideo properties:")
    print(f"  - Resolution: {data['video_properties']['width']}x{data['video_properties']['height']}")
    print(f"  - FPS: {data['video_properties']['fps']}")
    print(f"  - Total video frames: {data['video_properties']['frame_count']}")
    
    # Analyze first frame
    first_frame = data['frames'][0]
    print(f"\nFirst frame (frame {first_frame['frame_id']}):")
    print(f"  Target position: ({first_frame['target_position'][0]:.2f}, {first_frame['target_position'][1]:.2f})")
    print(f"  Number of interactions: {len(first_frame['interactions'])}")
    
    if first_frame['interactions']:
        # Sort by threat score
        sorted_interactions = sorted(first_frame['interactions'], key=lambda x: x['score'], reverse=True)
        print(f"\n  Top 5 threats:")
        for i, interaction in enumerate(sorted_interactions[:5], 1):
            print(f"    {i}. Object {interaction['object_id']}: score={interaction['score']:.3f}")
            print(f"       Position: ({interaction['position'][0]:.2f}, {interaction['position'][1]:.2f})")
            print(f"       Features: f1={interaction['features']['f1_distance']:.3f}, "
                  f"f2={interaction['features']['f2_velocity_diff']:.3f}, "
                  f"f3={interaction['features']['f3_heading_alignment']:.3f}, "
                  f"f4={interaction['features']['f4_class_interaction']:.3f}")
    
    # Analyze middle frame
    mid_frame_idx = len(data['frames']) // 2
    mid_frame = data['frames'][mid_frame_idx]
    print(f"\nMiddle frame (frame {mid_frame['frame_id']}):")
    print(f"  Target position: ({mid_frame['target_position'][0]:.2f}, {mid_frame['target_position'][1]:.2f})")
    print(f"  Number of interactions: {len(mid_frame['interactions'])}")
    
    if mid_frame['interactions']:
        sorted_interactions = sorted(mid_frame['interactions'], key=lambda x: x['score'], reverse=True)
        print(f"  Top threat: Object {sorted_interactions[0]['object_id']} with score {sorted_interactions[0]['score']:.3f}")
    
    # Statistics across all frames
    all_scores = []
    all_interaction_counts = []
    for frame in data['frames']:
        all_interaction_counts.append(len(frame['interactions']))
        for interaction in frame['interactions']:
            all_scores.append(interaction['score'])
    
    if all_scores:
        print(f"\nStatistics across all frames:")
        print(f"  Average interactions per frame: {sum(all_interaction_counts) / len(all_interaction_counts):.1f}")
        print(f"  Min interactions: {min(all_interaction_counts)}, Max interactions: {max(all_interaction_counts)}")
        print(f"  Average threat score: {sum(all_scores) / len(all_scores):.3f}")
        print(f"  Min threat score: {min(all_scores):.3f}, Max threat score: {max(all_scores):.3f}")
        print(f"  High threat count (>= 0.7): {sum(1 for s in all_scores if s >= 0.7)}")
        print(f"  Medium threat count (0.4-0.7): {sum(1 for s in all_scores if 0.4 <= s < 0.7)}")
        print(f"  Low threat count (< 0.4): {sum(1 for s in all_scores if s < 0.4)}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
    else:
        metadata_path = 'output/bookstore_video0_metadata.json'
    analyze_metadata(metadata_path)

