# Video Visualization Optimization Guide

This document outlines optimizations for threat score video visualization.

## 1. Video Codec Optimization

### Current: `mp4v` (MPEG-4 Part 2)
- **Pros**: Universal compatibility
- **Cons**: Large file sizes, older codec

### Recommended: H.264 (`avc1` or `H264`)
- **Pros**: Better compression (50-70% smaller files), modern standard, good quality
- **Cons**: Slightly slower encoding

### Alternative: H.265 (`hevc`)
- **Pros**: Even better compression (30-50% smaller than H.264)
- **Cons**: Less compatible, slower encoding

**Implementation**: Use `--codec avc1` or modify default in code.

## 2. Performance Optimizations

### A. Caching Coordinate Transformations
- Cache transformed coordinates for obstacles that don't move much
- Reduces redundant calculations

### B. Vectorized Operations
- Use NumPy vectorization for batch operations
- Process multiple obstacles simultaneously

### C. Parallel Frame Processing
- Use multiprocessing for independent frames
- Speed up: ~4-8x on multi-core systems

### D. Frame Skipping for Preview
- Option to process every Nth frame for quick previews
- Full quality render for final output

## 3. Visual Quality Optimizations

### A. Anti-aliasing
- Enable anti-aliasing for smoother lines and text
- Use `cv2.LINE_AA` (already implemented)

### B. Higher Resolution Rendering
- Option to render at 2x resolution then downscale
- Reduces aliasing artifacts

### C. Better Color Interpolation
- Use perceptually uniform color spaces (LAB/LUV)
- Smoother color transitions

### D. Adaptive Text Size
- Scale text based on zoom/distance
- Better readability

## 4. File Size Optimizations

### A. Variable Bitrate (VBR)
- Higher quality for complex frames
- Lower bitrate for simple frames

### B. Quality Settings
- `--quality high/medium/low` flags
- Adjusts bitrate and compression

### C. Resolution Scaling
- `--scale-down 0.5` for smaller files
- Maintains aspect ratio

## 5. Memory Optimizations

### A. Streaming Processing
- Process frames in batches
- Don't load entire video into memory

### B. Lazy Loading
- Only load frames when needed
- Release frames after processing

## 6. Recommended Settings

### For Best Quality (Large Files):
```bash
--codec avc1 --quality high
```

### For Balanced (Recommended):
```bash
--codec avc1 --quality medium
```

### For Small Files (Preview):
```bash
--codec avc1 --quality low --scale-down 0.75
```

### For Fast Processing:
```bash
--codec mp4v --frame-skip 2  # Process every 2nd frame
```

## 7. Implementation Status

✅ **Implemented:**
- Anti-aliased rendering (LINE_AA)
- Continuous color interpolation
- Efficient frame-by-frame processing

🔄 **Can Be Added:**
- H.264 codec option
- Quality presets
- Parallel processing
- Coordinate transformation caching

