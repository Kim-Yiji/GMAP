import os
import argparse
import glob
import numpy as np
from collections import defaultdict


SCENES = [
    "bookstore", "coupa", "deathCircle", "gates",
    "hyang", "little", "nexus", "quad",
]

# Exclusions per earlier requirements
EXCLUDE_VIDEOS = {
    "quad": ["video0", "video1", "video2", "video3"],
    "gates": ["video4", "video5", "video6", "video7", "video8"],
    "little": ["video0"],
    "nexus": ["video5", "video6", "video7"],
    "hyang": ["video7", "video8", "video9"],
    "deathCircle": ["video2", "video4"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./sdd_datasets", help="Root of raw SDD scene folders")
    parser.add_argument("--out_root", default="./dataset_out", help="Output root for processed splits")
    parser.add_argument("--mode", choices=["pedonly", "integrated"], default="pedonly",
                        help="Experiment mode: pedonly (train/test pedestrians only) or integrated (train all classes, test pedestrians only)")
    parser.add_argument("--test_scene", choices=SCENES, required=True, help="Scene held out for testing (LOSO)")
    parser.add_argument("--val_ratio_by_time", type=float, default=0.15, help="Validation fraction by total duration from train scenes")
    parser.add_argument("--min_val_seconds", type=float, default=90.0, help="Minimum total duration in seconds for validation")
    parser.add_argument("--fps", type=float, default=2.5, help="Approx FPS after annotation sampling (SDD effective fps ~2-3)")
    return parser.parse_args()


def load_annotations_file(path):
    # Expected SDD format per line:
    # frame id, track id, xmin, ymin, xmax, ymax, frame ?, lost, occluded, generated, label
    # We'll robustly parse and ignore malformed lines
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove quotes around label
            if line.count('"') >= 2:
                parts = line.split('"')
                label = parts[1]
                left = parts[0].strip()
                left_parts = left.split()
            else:
                toks = line.split()
                label = toks[-1].strip('"')
                left_parts = toks[:-1]
            try:
                nums = list(map(float, left_parts))
                if len(nums) < 10:
                    continue
                frame = int(nums[0])
                track_id = int(nums[1])
                xmin, ymin, xmax, ymax = nums[2], nums[3], nums[4], nums[5]
                lost = int(nums[-3])
                occluded = int(nums[-2])
                generated = int(nums[-1])
                x = (xmin + xmax) / 2.0
                y = (ymin + ymax) / 2.0
                data.append((frame, track_id, x, y, lost, occluded, generated, label))
            except Exception:
                continue
    if not data:
        return np.empty((0, 4)), []
    arr = np.array([[d[0], d[1], d[2], d[3]] for d in data], dtype=float)
    meta = [(d[0], d[1], d[4], d[5], d[6], d[7]) for d in data]
    return arr, meta


def filter_rows(arr, meta, include_generated=True):
    # Keep label info aligned
    kept = []
    kept_labels = []
    for i, m in enumerate(meta):
        frame, tid, lost, occluded, generated, label = m
        if lost != 0 or occluded != 0:
            continue
        if not include_generated and generated == 1:
            continue
        kept.append(arr[i])
        kept_labels.append(label)
    if not kept:
        return np.empty((0, 4)), []
    return np.stack(kept, axis=0), kept_labels


def class_filter(arr, labels, allowed_labels):
    if allowed_labels is None:
        return arr
    kept = [arr[i] for i, lab in enumerate(labels) if lab in allowed_labels]
    if not kept:
        return np.empty((0, 4))
    return np.stack(kept, axis=0)


def write_per_video_txt(out_dir, scene, video, class_suffix, data_rows):
    if data_rows.size == 0:
        return False
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{scene}_{video}_{class_suffix}.txt"
    out_path = os.path.join(out_dir, out_name)
    np.savetxt(out_path, data_rows, fmt="%d\t%d\t%.4f\t%.4f")
    return True


def collect_scene_videos(data_root, scene):
    scene_dir = os.path.join(data_root, scene)
    videos = []
    for vdir in sorted(os.listdir(scene_dir)):
        if EXCLUDE_VIDEOS.get(scene) and vdir in EXCLUDE_VIDEOS[scene]:
            continue
        ann = os.path.join(scene_dir, vdir, "annotations.txt")
        if os.path.isfile(ann):
            videos.append((vdir, ann))
    return videos


def seconds_of_rows(rows, fps):
    if rows.size == 0:
        return 0.0
    frames = rows[:, 0]
    # approximate by unique frames count
    uniq = len(np.unique(frames))
    return uniq / max(fps, 1e-6)


def main():
    args = parse_args()

    data_root = args.data_root
    out_root = args.out_root
    mode = args.mode
    test_scene = args.test_scene
    fps = args.fps

    # Define label sets
    ped_labels = {"Pedestrian", "Person"}
    all_labels = None  # integrated mode uses all

    # Prepare outputs
    exp_root = os.path.join(out_root, "dataset_pedonly" if mode == "pedonly" else "dataset_integrated")
    splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    # Determine train/val/test scenes
    train_scenes = [s for s in SCENES if s != test_scene]

    # First, load and pre-filter all per-video rows
    per_video_data = {}
    for scene in SCENES:
        per_video_data[scene] = []
        vids = collect_scene_videos(data_root, scene)
        for video, ann_path in vids:
            arr, meta = load_annotations_file(ann_path)
            arr_filt, labels = filter_rows(arr, meta, include_generated=True)
            if arr_filt.size == 0:
                continue
            # Class selection for storage; for integrated we'll store all; for pedonly we'll store only peds
            if mode == "pedonly":
                rows_to_store = class_filter(arr_filt, labels, ped_labels)
                class_suffix = "pedestrian"
            else:
                rows_to_store = arr_filt  # keep all classes in data
                class_suffix = "allclasses"
            per_video_data[scene].append((video, rows_to_store, class_suffix))

    # Assign test: all videos of test_scene
    for video, rows, class_suffix in per_video_data.get(test_scene, []):
        if rows.size == 0:
            continue
        write_per_video_txt(
            os.path.join(exp_root, test_scene, "test"),
            test_scene, video, class_suffix, rows.astype(float)
        )

    # Train + Val from remaining scenes; select val by total seconds
    # Greedy accumulate val videos until reaching target seconds and min seconds
    target_ratio = args.val_ratio_by_time
    min_val_sec = args.min_val_seconds
    # Collect candidate list with durations
    candidates = []
    for scene in train_scenes:
        for video, rows, class_suffix in per_video_data.get(scene, []):
            sec = seconds_of_rows(rows, fps)
            candidates.append((scene, video, rows, class_suffix, sec))

    total_train_time = sum(sec for _, _, _, _, sec in candidates)
    target_val_time = max(min_val_sec, total_train_time * target_ratio)
    candidates.sort(key=lambda x: -x[4])  # longest first for quick coverage

    val_time = 0.0
    val_selected = set()
    for scene, video, rows, class_suffix, sec in candidates:
        if val_time >= target_val_time and scene in train_scenes:
            continue
        # ensure at least one video per several scenes if possible by natural ordering
        val_selected.add((scene, video))
        val_time += sec

    # Write val/train
    for scene, video, rows, class_suffix, sec in candidates:
        subdir = "val" if (scene, video) in val_selected else "train"
        if rows.size == 0:
            continue
        write_per_video_txt(
            os.path.join(exp_root, scene, subdir),
            scene, video, class_suffix, rows.astype(float)
        )

    print(f"Done. Outputs at: {exp_root}")


if __name__ == "__main__":
    main()


