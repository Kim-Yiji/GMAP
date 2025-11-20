import os
import csv

# Default sizes used when no class_size_csv is provided or class not found
DEFAULT_CLASS_SIZES = {
    "Pedestrian": 1.0,
    "Skater": 1.2,
    "Biker": 1.5,
    "Cart": 2.0,
    "Car": 6.0,
    "Bus": 10.0,
}


def load_track_labels(labels_dir, basename):
    """Load track_id->class_name mapping for a given video basename.

    Expects a file named '<basename>_labels.csv' with header: track_id,label
    Returns: dict[int->str]
    """
    path = os.path.join(labels_dir, f"{basename}_labels.csv")
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row['track_id'])
            except Exception:
                continue
            mapping[tid] = row.get('label', 'Pedestrian')
    return mapping


def load_class_sizes(class_size_csv):
    """Load class_name->size mapping from CSV with header: label,size.
    Returns: dict[str->float]
    """
    sizes = {}
    if not class_size_csv or not os.path.exists(class_size_csv):
        # Fallback to built-in defaults
        return dict(DEFAULT_CLASS_SIZES)
    with open(class_size_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get('label')
            try:
                size = float(row.get('size', '0'))
            except Exception:
                size = 0.0
            if label:
                sizes[label] = size
    # Merge with defaults for any missing classes
    merged = dict(DEFAULT_CLASS_SIZES)
    merged.update(sizes)
    return merged


