"""
Streamlit frontend for Threat Score Visualization

This app allows users to upload videos and annotations, then generate
visualized threat score videos with interactive controls.
"""

import streamlit as st
import os
import tempfile
import numpy as np
from pathlib import Path
import json
import cv2

# Import threat score visualization modules
from threat_score_viz.visualizer import (
    process_video_with_threat_scores,
    transform_coordinates,
    calculate_coordinate_transform,
    draw_target_marker
)
from threat_score_viz.video_utils import get_video_properties, load_video_frame
from threat_score_viz.target_selector import find_central_target_candidates, select_target
from threat_score_viz.data_parser import load_annotations, get_object_positions_per_frame, get_available_object_ids

# Page configuration
st.set_page_config(
    page_title="Threat Score Visualizer",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Threat Score Visualizer")
st.markdown("Upload a video and annotations to generate a threat score visualization with nodes and edges.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Threat score parameters
    st.subheader("Threat Score Parameters")
    tau = st.slider("Tau (sigmoid temperature)", 0.05, 0.5, 0.15, 0.01)
    beta = st.slider("Beta (sigmoid midpoint)", 0.0, 1.0, 0.5, 0.01)
    ttc_max = st.slider("Max TTC (frames)", 5.0, 50.0, 20.0, 1.0)
    
    # Weights
    st.subheader("Feature Weights")
    w_distance = st.slider("Distance weight", 0.0, 1.0, 0.5, 0.05)
    w_velocity = st.slider("Approach velocity weight", 0.0, 1.0, 0.25, 0.05)
    w_size = st.slider("Obstacle size weight", 0.0, 1.0, 0.15, 0.05)
    w_ttc = st.slider("TTC weight", 0.0, 1.0, 0.1, 0.05)
    
    # Normalize weights
    total_weight = w_distance + w_velocity + w_size + w_ttc
    if total_weight > 0:
        w_distance /= total_weight
        w_velocity /= total_weight
        w_size /= total_weight
        w_ttc /= total_weight
    
    # FOV settings
    st.subheader("Field of View")
    use_fov = st.checkbox("Enable FOV filtering", value=True)
    fov_angle = st.slider("FOV angle (degrees)", 60, 180, 110, 5) if use_fov else None

# Main content area
with st.container():
    # Initialize variables
    tmp_video_path = None
    tmp_annot_path = None
    
    # Frame range configuration at the top
    st.subheader("📐 Frame Range Configuration")
    col_frame1, col_frame2 = st.columns(2)
    
    with col_frame1:
        process_all = st.checkbox("Process all frames", value=False)
    
    with col_frame2:
        if not process_all:
            start_frame = st.number_input("Start frame", 0, 10000, 0, 50, key="start_frame")
            end_frame = st.number_input("End frame", 0, 10000, 150, 50, key="end_frame")
            if end_frame <= start_frame:
                st.warning("End frame must be greater than start frame")
                end_frame = start_frame + 150
        else:
            start_frame = 0
            end_frame = None
    
    st.divider()
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Video")
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload the video file to visualize"
        )
    
    with col2:
        st.subheader("📄 Upload Annotations")
        annotation_file = st.file_uploader(
            "Choose an annotaion file",
            type=['txt'],
            help="Upload the corresponding annotation file"
        )
    
    # Display video info if uploaded
    if video_file is not None:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            tmp_video_path = tmp_video.name
        
        try:
            video_props = get_video_properties(tmp_video_path)
            st.info(f"""
            **Video Properties:**
            - Resolution: {video_props['width']}x{video_props['height']}
            - FPS: {video_props['fps']:.2f}
            - Frames: {video_props['frame_count']}
            - Duration: {video_props['duration']:.2f}s
            """)
            
            # Update frame range if processing all
            if 'process_all' in locals() and process_all:
                end_frame = video_props['frame_count'] - 1
        except Exception as e:
            st.error(f"Error reading video: {e}")
            tmp_video_path = None
    else:
        tmp_video_path = None
    
    # Display annotation info if uploaded
    if annotation_file is not None:
        # Save uploaded annotation to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_annot:
            tmp_annot.write(annotation_file.read().decode('utf-8'))
            tmp_annot_path = tmp_annot.name
        
        try:
            annotations_info = load_annotations(tmp_annot_path)
            unique_objects = len(set(annotations_info[:, 1].astype(int)))
            frame_range = (int(annotations_info[:, 0].min()), int(annotations_info[:, 0].max()))
            
            st.info(f"""
            **Annotation Properties:**
            - Total entries: {len(annotations_info)}
            - Unique objects: {unique_objects}
            - Frame range: {frame_range[0]} - {frame_range[1]}
            """)
        except Exception as e:
            st.error(f"Error reading annotations: {e}")
            tmp_annot_path = None
    
    # Target selection section
    selected_target_id = None
    preview_frame = None
    
    # Load annotations once if available (for target selection)
    annotations_loaded = None
    if tmp_annot_path is not None:
        try:
            annotations_loaded = load_annotations(tmp_annot_path)
        except:
            pass
    
    if tmp_video_path is not None and tmp_annot_path is not None:
        st.divider()
        st.subheader("🎯 Target Selection")
        
        # Get frame range from configuration (defined earlier)
        # Make sure start_frame and end_frame are available
        if 'start_frame' not in locals():
            start_frame = 0
        if 'end_frame' not in locals():
            end_frame = None
        
        # Generate preview frame
        try:
            video_props = get_video_properties(tmp_video_path)
            frame_data = get_object_positions_per_frame(tmp_annot_path)
            
            # Find a frame for preview - use start_frame if set, otherwise find a good frame
            preview_frame_id = None
            
            # If start_frame is set and within the frame data, use it
            if 'start_frame' in locals() and start_frame is not None:
                if start_frame in frame_data and len(frame_data[start_frame]) > 0:
                    preview_frame_id = start_frame
                else:
                    # Start frame doesn't have data, find closest frame with data
                    available_frames = sorted(frame_data.keys())
                    for fid in available_frames:
                        if fid >= start_frame:
                            preview_frame_id = fid
                            break
                    # If no frame after start_frame, use the last available frame
                    if preview_frame_id is None and available_frames:
                        preview_frame_id = available_frames[-1]
            
            # Fallback: find a frame with multiple objects
            if preview_frame_id is None:
                for fid in sorted(frame_data.keys())[:100]:  # Check first 100 frames
                    if len(frame_data[fid]) > 1:  # Frame with multiple objects
                        preview_frame_id = fid
                        break
            
            # Final fallback
            if preview_frame_id is None:
                preview_frame_id = sorted(frame_data.keys())[0] if frame_data else 0
            
            # Load preview frame
            success, frame = load_video_frame(tmp_video_path, preview_frame_id)
            if success and frame is not None:
                # Calculate coordinate transform - use same logic as video processing
                # This ensures preview matches the actual video output
                coord_transform = None
                
                # Try to use homography if available (same as video processing)
                homography_path = None
                scene_name = None
                video_number = 0
                
                import os
                annotation_dir = os.path.dirname(os.path.abspath(tmp_annot_path))
                annotation_filename = os.path.basename(tmp_annot_path)
                
                # Common scene names in SDD
                scene_names = ['bookstore', 'deathCircle', 'gates', 'hyang', 'nexus', 'quad', 'coupa', 'little']
                for scene in scene_names:
                    if scene.lower() in annotation_dir.lower() or scene.lower() in annotation_filename.lower():
                        scene_name = scene.lower()
                        # Try to extract video number from filename
                        import re
                        video_match = re.search(r'video(\d+)', annotation_filename, re.IGNORECASE)
                        if video_match:
                            video_number = int(video_match.group(1))
                        break
                
                # Look for homography file
                homography_candidates = [
                    'H_SDD.txt',
                    os.path.join(os.path.dirname(tmp_annot_path), '..', '..', 'H_SDD.txt'),
                    os.path.join(os.path.dirname(tmp_annot_path), 'H_SDD.txt'),
                ]
                for candidate in homography_candidates:
                    if os.path.exists(candidate):
                        homography_path = candidate
                        break
                
                # Use homography if available
                if homography_path and scene_name:
                    try:
                        from threat_score_viz.homography_parser import calculate_transform_from_homography
                        annotations = load_annotations(tmp_annot_path)
                        x_coords = annotations[:, 2].astype(float)
                        y_coords = annotations[:, 3].astype(float)
                        
                        coord_transform = calculate_transform_from_homography(
                            homography_path, scene_name, video_number,
                            video_props['width'], video_props['height'],
                            x_coords.min(), x_coords.max(),
                            y_coords.min(), y_coords.max()
                        )
                    except Exception as e:
                        pass  # Fall through to standard transformation
                
                # Fall back to standard transformation with offset adjustments (same as video processing)
                if coord_transform is None:
                    base_transform = calculate_coordinate_transform(
                        tmp_annot_path,
                        video_props['width'],
                        video_props['height'],
                        preserve_position=True
                    )
                    base_scale_x, base_scale_y, base_offset_x, base_offset_y = base_transform
                    
                    # Apply same default offset adjustments as video processing
                    offset_adjustment_x = -90  # Default: shift left 90px
                    offset_adjustment_y = 100   # Default: shift down 100px
                    
                    adjusted_offset_x = base_offset_x + offset_adjustment_x
                    adjusted_offset_y = base_offset_y + offset_adjustment_y
                    
                    coord_transform = (base_scale_x, base_scale_y, adjusted_offset_x, adjusted_offset_y)
                
                # Draw all objects on preview frame
                preview_frame = frame.copy()
                objects_in_frame = frame_data.get(preview_frame_id, [])
                available_ids = []
                
                for ped_id, x, y in objects_in_frame:
                    available_ids.append(int(ped_id))
                    # Transform coordinates
                    scale_x, scale_y, offset_x, offset_y = coord_transform
                    pixel_x, pixel_y = transform_coordinates(x, y, scale_x, scale_y, offset_x, offset_y)
                    px, py = int(pixel_x), int(pixel_y)
                    
                    # Draw circle for each object
                    cv2.circle(preview_frame, (px, py), 8, (0, 255, 0), 2)  # Green circle
                    # Draw ID label
                    id_text = f"ID:{ped_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(id_text, font, font_scale, thickness)
                    # Background rectangle for text
                    cv2.rectangle(
                        preview_frame,
                        (px - 5, py - text_height - 5),
                        (px + text_width + 5, py + 5),
                        (0, 0, 0),
                        -1
                    )
                    # Text
                    cv2.putText(
                        preview_frame,
                        id_text,
                        (px, py),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )
                
                # Convert BGR to RGB for display
                preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                
                # Get all available object IDs (not just in this frame)
                if annotations_loaded is not None:
                    annotations = annotations_loaded
                else:
                    annotations = load_annotations(tmp_annot_path)
                all_ids = get_available_object_ids(annotations)
                
                # Don't auto-select - let user choose
                # Initialize selected_target_id from session state (starts as None)
                if 'selected_target_id' not in st.session_state:
                    st.session_state.selected_target_id = None
                
                # Also track if processing should happen
                if 'process_video' not in st.session_state:
                    st.session_state.process_video = False
                
                # Show object IDs with frame counts
                object_info = {}
                for obj_id in all_ids:
                    obj_frames = annotations[annotations[:, 1].astype(int) == obj_id]
                    frame_count = len(obj_frames)
                    object_info[obj_id] = frame_count
                
                # Compact target selection with better layout
                
                # Show info about which frame is being previewed
                frame_range_info = ""
                if start_frame is not None and end_frame is not None:
                    frame_range_info = f" (Frame range: {start_frame}-{end_frame})"
                elif start_frame is not None:
                    frame_range_info = f" (Starting at frame {start_frame})"
                
                # Create three columns: preview, buttons, info
                col_preview, col_buttons, col_info = st.columns([3, 2, 2])
                
                with col_preview:
                    # Highlight selected target in preview (create a copy for highlighting)
                    preview_with_highlight = preview_rgb.copy()
                    if st.session_state.selected_target_id and st.session_state.selected_target_id in available_ids:
                        # Find position of selected target
                        for ped_id, x, y in objects_in_frame:
                            if int(ped_id) == st.session_state.selected_target_id:
                                scale_x, scale_y, offset_x, offset_y = coord_transform
                                pixel_x, pixel_y = transform_coordinates(x, y, scale_x, scale_y, offset_x, offset_y)
                                px, py = int(pixel_x), int(pixel_y)
                                
                                # Draw highlight (larger circle) - red in RGB
                                cv2.circle(preview_with_highlight, (px, py), 20, (255, 0, 0), 4)  # Red highlight
                                # Also draw a thicker outer ring
                                cv2.circle(preview_with_highlight, (px, py), 25, (255, 255, 0), 2)  # Yellow outer ring
                                break
                    
                    st.caption(f"Preview: Frame {preview_frame_id}{frame_range_info}")
                    st.image(preview_with_highlight, use_container_width=True, channels="RGB")
                    if st.session_state.selected_target_id and st.session_state.selected_target_id in available_ids:
                        st.caption("🔴 Selected | 🟢 Others")
                    else:
                        st.caption("🟢 Click a button →")
                
                with col_buttons:
                    st.markdown("**Quick Select:**")
                    
                    # Sort people in frame by ID
                    people_in_frame = sorted([int(ped_id) for ped_id, _, _ in objects_in_frame])
                    
                    if people_in_frame:
                        # Create compact buttons in a grid (2 columns)
                        button_cols = st.columns(2)
                        for idx, person_id in enumerate(people_in_frame):
                            frame_count = object_info.get(person_id, 0)
                            is_selected = (st.session_state.selected_target_id == person_id)
                            
                            # Use alternating columns
                            with button_cols[idx % 2]:
                                button_label = f"ID {person_id}"
                                if is_selected:
                                    button_label = f"✅ {button_label}"
                                
                                if st.button(
                                    button_label,
                                    key=f"person_btn_{person_id}",
                                    use_container_width=True,
                                    type="primary" if is_selected else "secondary"
                                ):
                                    st.session_state.selected_target_id = person_id
                                    st.session_state.process_video = False
                                    st.rerun()
                                
                                # Compact frame count
                                st.caption(f"{frame_count} frames", help=f"Person ID {person_id}")
                        
                        st.markdown("---")
                        st.markdown("**Or search all:**")
                        
                        # Compact dropdown
                        sorted_ids = sorted(all_ids, key=lambda x: object_info[x], reverse=True)
                        options = ["(Select)"] + [f"ID {obj_id} ({object_info[obj_id]}f)" for obj_id in sorted_ids]
                        option_to_id = {opt: obj_id for opt, obj_id in zip(options[1:], sorted_ids)}
                        
                        current_option = None
                        if st.session_state.selected_target_id and st.session_state.selected_target_id in sorted_ids:
                            current_option = f"ID {st.session_state.selected_target_id} ({object_info[st.session_state.selected_target_id]}f)"
                        
                        selected_option = st.selectbox(
                            "All people:",
                            options=options,
                            index=0 if current_option is None else options.index(current_option),
                            help="Select any person",
                            key="target_selectbox",
                            label_visibility="collapsed"
                        )
                        
                        if selected_option != "(Select)":
                            selected_id_from_dropdown = option_to_id[selected_option]
                            if selected_id_from_dropdown != st.session_state.selected_target_id:
                                st.session_state.selected_target_id = selected_id_from_dropdown
                                st.session_state.process_video = False
                                st.rerun()
                    else:
                        st.warning("No people in frame")
                
                with col_info:
                    if st.session_state.selected_target_id:
                        st.success(f"✅ **Selected**")
                        st.metric("Person ID", st.session_state.selected_target_id)
                        frame_count = object_info.get(st.session_state.selected_target_id, 0)
                        st.metric("Frames", frame_count)
                        
                        if frame_count < 10:
                            st.warning(f"⚠️ Only {frame_count} frames")
                        elif frame_count < 50:
                            st.info(f"ℹ️ {frame_count} frames available")
                        else:
                            st.success(f"✓ {frame_count} frames")
                    else:
                        st.info("👈 **Select a person**")
                        st.caption("Click a button or use dropdown")
                
                # Set the selected_target_id for use in processing
                selected_target_id = st.session_state.selected_target_id
            else:
                st.warning("Could not load preview frame")
        except Exception as e:
            st.warning(f"Could not generate preview: {e}")
            # Fallback: just show object list
            try:
                annotations = load_annotations(tmp_annot_path)
                all_ids = get_available_object_ids(annotations)
                if all_ids:
                    default_target_id, _ = select_target(
                        tmp_annot_path,
                        auto_select=True,
                        prefer_center=True,
                        min_frames=100
                    )
                    selected_target_id = st.selectbox(
                        "Select target person ID:",
                        options=all_ids,
                        index=all_ids.index(default_target_id) if default_target_id in all_ids else 0
                    )
            except:
                pass
    
    # Process button - only enabled if target is selected
    process_disabled = (tmp_video_path is None or tmp_annot_path is None or st.session_state.get('selected_target_id') is None)
    
    if st.button(
        "🚀 Generate Visualization Video", 
        type="primary", 
        use_container_width=True,
        disabled=process_disabled
    ):
        if tmp_video_path is None or tmp_annot_path is None:
            st.error("Please upload both video and annotation files")
        elif st.session_state.get('selected_target_id') is None:
            st.error("Please select a target person first by clicking on one of the ID buttons above")
        else:
            # Create progress container
            progress_container = st.container()
            status_text = progress_container.empty()
            
            try:
                status_text.info("📋 Step 1/4: Analyzing video and annotations...")
                
                # Create output directory
                output_dir = tempfile.mkdtemp()
                output_video_path = os.path.join(output_dir, "output_video.mp4")
                output_metadata_path = os.path.join(output_dir, "output_metadata.json")
                
                # Get video properties for target selection
                video_props = get_video_properties(tmp_video_path)
                
                status_text.info("🎯 Step 2/4: Validating selected target...")
                
                # Use user-selected target (required)
                target_id = st.session_state.selected_target_id
                if target_id is None:
                    st.error("No target selected. Please select a person first.")
                    st.stop()
                
                # Validate and get stats
                from threat_score_viz.target_selector import validate_target
                is_valid, target_stats = validate_target(tmp_annot_path, target_id)
                if not is_valid:
                    st.error(f"Selected target ID {target_id} is invalid or not found in annotations.")
                    st.stop()
                
                # Show target selection results
                with st.expander("🎯 Selected Target Information", expanded=True):
                    st.success(f"**Target: Person ID {target_id}**")
                    if 'avg_position' in target_stats:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Frames", target_stats['frame_count'])
                            st.metric("Distance from Center", f"{target_stats.get('distance_from_center', 0):.1f}px")
                        with col_b:
                            avg_pos = target_stats['avg_position']
                            st.metric("Average Position", f"({avg_pos[0]:.1f}, {avg_pos[1]:.1f})")
                            st.metric("Combined Score", f"{target_stats.get('combined_score', 0):.3f}")
                    else:
                        st.info(f"Appears in {target_stats['frame_count']} frames")
                
                status_text.info("⚙️ Step 3/4: Processing video frames...")
                
                # Convert FOV angle to radians
                fov_angle_rad = None
                if use_fov and fov_angle is not None:
                    fov_angle_rad = np.pi * fov_angle / 180.0
                
                # Process video (this may take a while)
                stats = process_video_with_threat_scores(
                    video_path=tmp_video_path,
                    annotation_path=tmp_annot_path,
                    output_video_path=output_video_path,
                    output_metadata_path=output_metadata_path,
                    target_id=target_id,
                    auto_select_target=False,
                    weights=(w_distance, w_velocity, w_size, w_ttc),
                    tau=tau,
                    beta=beta,
                    eps=1e-6,
                    ttc_max=ttc_max,
                    fov_angle=fov_angle_rad,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    scale=1.0,
                    draw_target=True,
                    draw_graph=True,
                    draw_edges=True,
                    draw_nodes=True,
                    apply_coordinate_transform=True
                )
                
                status_text.success("✅ Step 4/4: Video processing complete!")
                
                # Clear status
                progress_container.empty()
                
                st.success("🎉 **Visualization generated successfully!**")
                
                # Display results
                st.divider()
                st.header("🎬 Generated Video")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if os.path.exists(output_video_path):
                        # Get file size
                        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                        st.caption(f"File size: {file_size:.2f} MB | Target: Person ID {target_id}")
                        
                        # Read video as bytes for better compatibility
                        try:
                            with open(output_video_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            # Display video in Streamlit using bytes
                            st.video(video_bytes, format='video/mp4')
                            
                            # Download button below video
                            st.download_button(
                                label="📥 Download Video",
                                data=video_bytes,
                                file_name=f"threat_score_video_target_{target_id}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error loading video: {e}")
                            st.info("💡 Try downloading the video and playing it locally, or check the video file format.")
                            # Fallback: try with file path
                            try:
                                st.video(output_video_path)
                            except:
                                st.warning("Video preview not available. Please download the video to view it.")
                                # Still provide download
                                with open(output_video_path, 'rb') as f:
                                    video_bytes = f.read()
                                st.download_button(
                                    label="📥 Download Video",
                                    data=video_bytes,
                                    file_name=f"threat_score_video_target_{target_id}.mp4",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                    else:
                        st.error("Output video file not found")
                
                with col2:
                    st.subheader("📊 Processing Statistics")
                    st.json({
                        "Frames processed": stats['frames_processed'],
                        "Frames skipped": stats['frames_skipped'],
                        "Total frames": stats['total_frames'],
                        "Target ID": target_id,
                        "Processing rate": f"{stats['frames_processed'] / max(stats['total_frames'], 1) * 100:.1f}%"
                    })
                    
                    if os.path.exists(output_metadata_path):
                        # Get metadata file size
                        metadata_size = os.path.getsize(output_metadata_path) / 1024  # KB
                        st.caption(f"Metadata size: {metadata_size:.2f} KB")
                        
                        with open(output_metadata_path, 'rb') as f:
                            metadata_bytes = f.read()
                            st.download_button(
                                label="📥 Download Metadata (JSON)",
                                data=metadata_bytes,
                                file_name=f"threat_score_metadata_target_{target_id}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    else:
                        st.warning("Metadata file not found")
                    
            except Exception as e:
                status_text.error(f"❌ Error: {str(e)}")
                st.error(f"**Error processing video:** {str(e)}")
                import traceback
                with st.expander("🔍 Error Details (Click to expand)"):
                    st.code(traceback.format_exc())
            finally:
                # Cleanup temp files
                try:
                    if tmp_video_path and os.path.exists(tmp_video_path):
                        os.unlink(tmp_video_path)
                    if tmp_annot_path and os.path.exists(tmp_annot_path):
                        os.unlink(tmp_annot_path)
                except:
                    pass

