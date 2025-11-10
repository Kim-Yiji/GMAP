# Graph Visualization Feature

## Overview

The threat score visualization system now includes graph-like visualization that shows:
1. **Target Person**: Clearly identified with prominent marker and ID
2. **Graph Structure**: Nodes (target + obstacles) and edges (threat scores) connecting them

## Visualization Features

### 1. Target Person Identification
- **Prominent Marker**: Large, distinctive marker (cyan outer ring + magenta fill)
- **Clear Labeling**: "TARGET ID:XXX" label with highlighted background
- **Visibility**: Always drawn on top of other elements

### 2. Graph Structure

#### Nodes
- **Target Node**: Large, prominent marker (cyan/magenta)
- **Obstacle Nodes**: Colored circles representing each obstacle
  - Node size: Based on threat score (higher threat = larger node)
  - Node color: Based on threat score
    - Red: High threat (>= 0.7)
    - Orange/Yellow: Medium threat (0.4-0.7)
    - Green/Yellow: Low threat (< 0.4)
  - Node labels: "ID:XXX" showing obstacle ID

#### Edges
- **Connection Lines**: Lines connecting target to each obstacle
- **Edge Thickness**: Based on threat score (1-5 pixels)
  - Higher threat = thicker line
- **Edge Color**: Same color scheme as nodes
  - Red: High threat
  - Orange/Yellow: Medium threat
  - Green/Yellow: Low threat
- **Edge Labels**: Threat score displayed along the edge (midpoint)

### 3. Color Coding

The visualization uses a color gradient to represent threat levels:

- **High Threat (>= 0.7)**: Red
- **Medium Threat (0.4-0.7)**: Orange to Yellow (interpolated)
- **Low Threat (< 0.4)**: Green to Yellow (interpolated)

## Usage

### Basic Usage (with graph visualization)

```bash
python3 -m threat_score_viz \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/video_graph.mp4 \
  --output-metadata output/metadata.json \
  --target-id 176
```

### Disable Graph Visualization (use simple annotations)

```bash
python3 -m threat_score_viz \
  --video sdd_bookstore/bookstore_vid/video0/video.mp4 \
  --annotations sdd_bookstore/test/bookstore_video0_test.txt \
  --output-video output/video_simple.mp4 \
  --output-metadata output/metadata.json \
  --target-id 176 \
  --no-graph
```

### Customize Graph Elements

```bash
# Disable edges (only show nodes)
python3 -m threat_score_viz ... --no-edges

# Disable nodes (only show edges)
python3 -m threat_score_viz ... --no-nodes

# Disable target marker
python3 -m threat_score_viz ... --no-target-marker
```

## Graph Structure

The visualization represents a graph where:
- **Vertices (Nodes)**:
  - Target person (center node)
  - Obstacles (cars, pedestrians, bicycles, etc.)
- **Edges**:
  - Connections between target and obstacles
  - Edge weight = threat score
  - Visualized as line thickness and color

## Example Output

The graph visualization shows:
1. Target person (ID: 176) with prominent marker
2. Multiple obstacle nodes (colored circles with IDs)
3. Edges connecting target to obstacles (colored lines with threat scores)
4. Threat scores displayed along edges and on nodes

This creates an intuitive graph-like representation where:
- **Node size** indicates threat level
- **Edge thickness** indicates threat level
- **Color** indicates threat level (red = high, green = low)
- **Spatial layout** shows actual positions in the scene

## Benefits

1. **Clear Target Identification**: Easy to identify which person is being tracked
2. **Visual Relationships**: Graph structure shows relationships between target and obstacles
3. **Threat Assessment**: Color and size coding make threat levels immediately visible
4. **Spatial Context**: Graph overlaid on video maintains spatial relationships
5. **Scalable**: Works with any number of obstacles

## Technical Details

### Drawing Order
1. Edges are drawn first (background layer)
2. Target node is drawn (most prominent)
3. Obstacle nodes are drawn (on top of edges)

### Coordinate System
- All positions use image coordinates (x, y)
- Nodes are drawn at object positions
- Edges connect target to obstacle positions
- Text labels are positioned to avoid overlap

### Performance
- Graph visualization adds minimal overhead
- Processing speed: ~20-30 frames/second
- Suitable for real-time or batch processing

