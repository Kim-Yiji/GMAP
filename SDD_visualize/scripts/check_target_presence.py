#!/usr/bin/env python3
"""Check whether target track_id 1 exists in scored CSV and annotations.
Print a concise summary: frames present in scored CSV, frames present in annotations for track 1,
and sample positions.
"""
import csv
from collections import defaultdict

SCORED = 'outputs/pedestrian_1_scored.csv'
ANN = 'SDD_datasets/SDD_raw/coupa/video0/annotations.csv'
TARGET = 1

# read scored csv
frames_scores = defaultdict(list)
target_ids = set()
with open(SCORED, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        fr = int(row['frame'])
        try:
            t = int(row.get('target_track_id') or row.get('target') or TARGET)
        except Exception:
            t = TARGET
        target_ids.add(t)
        cx = float(row.get('target_cx') or 0)
        cy = float(row.get('target_cy') or 0)
        frames_scores[fr].append({'target':t, 'cx':cx, 'cy':cy, 'other': row.get('other_track_id')})

score_frames = sorted(frames_scores.keys())

# read annotations for track 1
ann_frames = set()
with open(ANN, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            tid = int(row.get('track_id') or row.get('track') or 0)
            fr = int(row.get('frame') or 0)
        except Exception:
            continue
        if tid == TARGET:
            ann_frames.add(fr)

# summary
print('Scored CSV target_track_ids found:', sorted(target_ids))
if score_frames:
    print('Scored CSV frames: count=', len(score_frames), 'min=', score_frames[0], 'max=', score_frames[-1])
    # sample first/mid/last
    first = score_frames[0]
    mid = score_frames[len(score_frames)//2]
    last = score_frames[-1]
    def sample_pos(fr):
        recs = frames_scores[fr]
        # take first rec
        r = recs[0]
        return r['cx'], r['cy']
    print('Sample positions:')
    print('  first', first, sample_pos(first))
    print('  mid', mid, sample_pos(mid))
    print('  last', last, sample_pos(last))
else:
    print('No frames in scored CSV')

print('\nAnnotations: track', TARGET, 'frames count=', len(ann_frames), end='')
if ann_frames:
    afs = sorted(ann_frames)
    print(' min=', afs[0], 'max=', afs[-1])
else:
    print()

# compare
common = set(score_frames) & ann_frames
print('\nFrames where both scored CSV and annotation for target exist:', len(common))
if len(common) <= 20:
    print('  list:', sorted(common))
else:
    cf = sorted(common)
    print('  examples:', cf[:5], '...', cf[-5:])

# frames in scored CSV but missing annotation
missing_ann = [fr for fr in score_frames if fr not in ann_frames]
print('\nFrames present in scored CSV but missing annotation bbox for target:', len(missing_ann))
if len(missing_ann) <= 20:
    print('  list:', missing_ann)
else:
    print('  examples:', missing_ann[:5], '...', missing_ann[-5:])

# done
print('\nDone')
