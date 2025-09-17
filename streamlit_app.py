"""Streamlit web interface for Action Shot Extractor."""

import streamlit as st
import tempfile
import zipfile
import os
from pathlib import Path
import shutil
from PIL import Image
import time

# Configure page
st.set_page_config(
    page_title="Action Shot Extractor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the pipeline
try:
    from src.action_shot_extractor.pipeline import run_pipeline
    from src.action_shot_extractor.player_matcher import create_player_matcher
except ImportError:
    st.error("Action Shot Extractor package not found. Please install the package first.")
    st.stop()

def main():
    """Main Streamlit application."""

    st.title("⚽ Action Shot Extractor")
    st.markdown("**AI-powered sports action shot extraction from video**")

    # Sidebar for configuration
    st.sidebar.header("🎯 Detection Method")

    detection_method = st.sidebar.radio(
        "Choose detection method:",
        ["Color-based", "Reference frames"],
        help="Color-based works when your player wears unique colors. Reference frames work for identical uniforms."
    )

    # Detection method specific inputs
    detection_config = {}

    if detection_method == "Color-based":
        st.sidebar.subheader("🎨 Color Settings")
        hex_color = st.sidebar.color_picker(
            "Player color (unique gear/jersey)",
            value="#FF0000",
            help="Pick the dominant color that distinguishes your target player"
        )
        detection_config = {"hex_color": hex_color}

    else:  # Reference frames
        st.sidebar.subheader("📸 Reference Images")
        st.sidebar.markdown("Upload 3 images of your target player:")

        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            front_img = st.file_uploader("Front view", type=['jpg', 'jpeg', 'png'], key="front")
        with col2:
            side_img = st.file_uploader("Side view", type=['jpg', 'jpeg', 'png'], key="side")
        with col3:
            back_img = st.file_uploader("Back view", type=['jpg', 'jpeg', 'png'], key="back")

        if front_img and side_img and back_img:
            # Show preview of uploaded images
            st.sidebar.markdown("**Preview:**")
            preview_cols = st.sidebar.columns(3)
            for i, (img, name) in enumerate([(front_img, "Front"), (side_img, "Side"), (back_img, "Back")]):
                with preview_cols[i]:
                    st.image(Image.open(img), caption=name, width=80)

    # Advanced settings
    st.sidebar.header("⚙️ Advanced Settings")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        workers = st.number_input("Parallel workers", min_value=1, max_value=8, value=4, help="More workers = faster processing")
        max_frames = st.number_input("Max frames to process", min_value=10, max_value=10000, value=1000, step=50, help="Limit frames for faster testing")

    with col2:
        blur_threshold = st.number_input("Sharpness threshold", min_value=50.0, max_value=500.0, value=150.0, step=10.0, help="Higher = only very sharp frames")
        confidence = st.number_input("Object detection confidence", min_value=0.1, max_value=1.0, value=0.35, step=0.05, help="Lower = detect more objects")

    # Main content area
    st.header("📹 Video Upload")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload your sports video (recommended: under 100MB for faster processing)"
    )

    if uploaded_video is not None:
        # Show video info
        st.success(f"✅ Video uploaded: {uploaded_video.name} ({uploaded_video.size / (1024*1024):.1f} MB)")

        # Processing button
        process_button = st.button("🚀 Extract Action Shots", type="primary", use_container_width=True)

        if process_button:
            # Validate inputs
            if detection_method == "Reference frames" and not (front_img and side_img and back_img):
                st.error("❌ Please upload all 3 reference images (front, side, back)")
                return

            # Create temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Save uploaded video
                video_path = temp_path / uploaded_video.name
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                # Handle reference frames if needed
                if detection_method == "Reference frames":
                    ref_dir = temp_path / "reference_frames"
                    ref_dir.mkdir()

                    # Save reference images
                    for img, filename in [(front_img, "front.jpg"), (side_img, "side.jpg"), (back_img, "back.jpg")]:
                        img_path = ref_dir / filename
                        with open(img_path, "wb") as f:
                            f.write(img.getbuffer())

                    detection_config = {"reference_dir": ref_dir}

                # Output directory
                output_dir = temp_path / "output"
                output_dir.mkdir()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Update status
                    status_text.text("🔄 Processing video...")
                    progress_bar.progress(10)

                    # Run the pipeline
                    summary = run_pipeline(
                        video=video_path,
                        detection_method=detection_method.lower().replace("-", "_").replace(" ", "_"),
                        detection_config=detection_config,
                        output_dir=output_dir,
                        max_workers=workers,
                        max_frames=max_frames,
                        blur_thr=blur_threshold,
                        conf=confidence,
                        dry_run=False
                    )

                    progress_bar.progress(90)
                    status_text.text("📊 Processing results...")

                    # Display results
                    st.success(f"🎉 Processing complete!")

                    # Results summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Frames Processed", summary.frames_processed)
                    with col2:
                        st.metric("Action Shots Found", len(summary.hits))
                    with col3:
                        hit_rate = (len(summary.hits) / summary.frames_processed * 100) if summary.frames_processed > 0 else 0
                        st.metric("Hit Rate", f"{hit_rate:.1f}%")
                    with col4:
                        st.metric("Detection Method", detection_method)

                    progress_bar.progress(100)
                    status_text.text("✅ Complete! Showing results below.")

                    # Display extracted frames
                    if summary.hits:
                        st.header("🏆 Extracted Action Shots")

                        # Create download package
                        zip_path = temp_path / "action_shots.zip"
                        with zipfile.ZipFile(zip_path, 'w') as zip_file:
                            for hit in summary.hits:
                                if hit.export_path and hit.export_path.exists():
                                    zip_file.write(hit.export_path, hit.export_path.name)

                        # Download button
                        with open(zip_path, "rb") as zip_file:
                            st.download_button(
                                label="📦 Download All Action Shots (ZIP)",
                                data=zip_file.read(),
                                file_name=f"action_shots_{uploaded_video.name.split('.')[0]}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                        # Display thumbnails
                        st.subheader("Preview (first 12 shots)")
                        cols = st.columns(4)

                        for i, hit in enumerate(summary.hits[:12]):
                            with cols[i % 4]:
                                if hit.export_path and hit.export_path.exists():
                                    try:
                                        img = Image.open(hit.export_path)
                                        st.image(img, caption=f"Frame {hit.index}", use_column_width=True)

                                        # Frame details
                                        st.caption(f"Sharpness: {hit.sharpness:.0f}")
                                        st.caption(f"Objects: {hit.detection_count}")
                                    except Exception as e:
                                        st.error(f"Error loading image: {e}")

                        if len(summary.hits) > 12:
                            st.info(f"📸 Showing 12 of {len(summary.hits)} total action shots. Download the ZIP file to get all frames.")

                    else:
                        st.warning("🤔 No action shots found. Try adjusting the settings:")
                        st.markdown("""
                        - Lower the **sharpness threshold** to include more frames
                        - Lower the **confidence threshold** to detect more objects
                        - Try a different **detection method**
                        - Check if your video contains the target player and ball
                        """)

                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("")
                    st.error(f"❌ Error processing video: {str(e)}")
                    st.markdown("**Troubleshooting tips:**")
                    st.markdown("""
                    - Ensure your video file is valid and not corrupted
                    - Try a smaller video file or reduce max frames
                    - For reference frames method, ensure images clearly show the player
                    - Check that the player and ball are visible in the video
                    """)

    # Sidebar help
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    **Color-based detection:**
    - Works best when your target player wears unique colors
    - Good for goalkeepers or players with distinctive gear

    **Reference frames method:**
    - Better for teams with identical uniforms
    - Upload clear images showing front, side, and back views
    - Works with different lighting and angles

    **Performance:**
    - More workers = faster processing
    - Reduce max frames for quicker testing
    - Higher thresholds = fewer but better quality shots
    """)

if __name__ == "__main__":
    main()