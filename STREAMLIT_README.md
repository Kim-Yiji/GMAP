# Streamlit Frontend for Threat Score Visualization

A web-based interface for generating threat score visualizations from videos and annotations.

## Installation

1. Install Streamlit and dependencies:
```bash
pip install -r requirements_streamlit.txt
```

Or install manually:
```bash
pip install streamlit opencv-python numpy
```

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 1. Video Upload
- Upload video files (MP4, AVI, MOV)
- Automatic video property detection (resolution, FPS, frame count)

### 2. Annotation Upload
- Upload annotation text files
- Automatic annotation statistics display

### 3. Interactive Configuration
- **Threat Score Parameters**:
  - Tau (sigmoid temperature): Controls threat score sensitivity
  - Beta (sigmoid midpoint): Controls threat score center point
  - Max TTC: Maximum time-to-collision value
  
- **Feature Weights**:
  - Distance weight
  - Approach velocity weight
  - Obstacle size weight
  - TTC weight
  - Weights are automatically normalized

- **Field of View**:
  - Enable/disable FOV filtering
  - Adjustable FOV angle (60-180 degrees)

- **Frame Range**:
  - Process all frames or specific range
  - Start and end frame selection

### 4. Automatic Target Selection

The app automatically selects the "well seen" person using:

**Combined Scoring (Centrality 70% + Frame Count 30%)**:
- **Centrality (70%)**: Prefers people near the center of the video
  - Main subjects are typically central
  - Better visual quality for analysis
  
- **Frame Count (30%)**: Prefers people visible in many frames
  - Indicates consistent presence
  - More stable for tracking
  - Important to the scene

**Why this works**:
- Balances "main subject" (central) with "stability" (high visibility)
- Avoids edge cases (edge person with many frames, or central person with few frames)
- Ensures the selected target is both prominent and trackable

### 5. Video Processing

- Real-time progress indication
- Automatic coordinate transformation
- Graph visualization (nodes and edges)
- Threat score color coding

### 6. Results

- **Download output video**: MP4 format with threat score visualization
- **Download metadata**: JSON file with detailed threat score data
- **Processing statistics**: Frame counts, target ID, etc.

## Usage Example

1. Click "Upload Video" and select your video file
2. Click "Upload Annotations" and select your annotation file
3. Adjust parameters in the sidebar (or use defaults)
4. Click "🚀 Generate Visualization"
5. Wait for processing to complete
6. Download the output video and metadata

## Technical Details

### Target Selection Algorithm

```python
# Combined score calculation
centrality_score = 1.0 - (normalized_distance_from_center)
frame_score = min(frame_count / 5000.0, 1.0)
combined_score = centrality_score * 0.7 + frame_score * 0.3
```

### Minimum Requirements

- Minimum frames: 100 (filters brief appearances)
- Center calculation: Relative center of annotation coordinate space
- Radius tolerance: 1.5× calculated center radius

## Troubleshooting

### "No suitable target found"
- Reduce minimum frames requirement
- Check that annotations contain valid data
- Verify video and annotation files match

### Video processing errors
- Ensure video file is valid and readable
- Check annotation file format matches SDD format
- Verify frame range is within video bounds

### Import errors
- Install all dependencies: `pip install -r requirements_streamlit.txt`
- Ensure you're in the correct directory
- Check Python version (3.8+ required)

