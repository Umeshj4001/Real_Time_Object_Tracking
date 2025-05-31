import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import base64
from tracker import EuclideanDistTracker
from streamlit.components.v1 import html

# Set page configuration for a professional look
st.set_page_config(
    page_title="Object Tracking Dashboard",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #4e657a;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .video-container {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to convert video file to base64
def video_to_base64(file_path):
    with open(file_path, "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode()

# App header
st.title("📹 Object Tracking Dashboard")
st.markdown("Upload a video to track objects using Euclidean Distance Tracking with OpenCV. Customize the Region of Interest (ROI) and view the results below.")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("Adjust the parameters for object tracking.")

# ROI parameters
st.sidebar.subheader("Region of Interest (ROI)")
roi_y_start = st.sidebar.slider("ROI Y-Start", 0, 1080, 340, help="Starting Y-coordinate for ROI")
roi_y_end = st.sidebar.slider("ROI Y-End", 0, 1080, 720, help="Ending Y-coordinate for ROI")
roi_x_start = st.sidebar.slider("ROI X-Start", 0, 1920, 500, help="Starting X-coordinate for ROI")
roi_x_end = st.sidebar.slider("ROI X-End", 0, 1920, 800, help="Ending X-coordinate for ROI")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a video file (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"],
    help="Select a video file to process"
)

if uploaded_file is not None:
    try:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            os.unlink(video_path)
            st.stop()

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate ROI
        if roi_y_end <= roi_y_start or roi_x_end <= roi_x_start:
            st.error("Invalid ROI: End coordinates must be greater than start coordinates.")
            cap.release()
            os.unlink(video_path)
            st.stop()
        if roi_y_end > height or roi_x_end > width:
            st.error(f"ROI exceeds video dimensions (width: {width}, height: {height}).")
            cap.release()
            os.unlink(video_path)
            st.stop()

        # Initialize tracker and detector
        tracker = EuclideanDistTracker()
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

        # Create temporary files for output videos
        tracked_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        roi_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        mask_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        tracked_out = cv2.VideoWriter(tracked_output_file, fourcc, fps, (width, height))
        roi_out = cv2.VideoWriter(roi_output_file, fourcc, fps, (roi_x_end - roi_x_start, roi_y_end - roi_y_start))
        mask_out = cv2.VideoWriter(mask_output_file, fourcc, fps, (roi_x_end - roi_x_start, roi_y_end - roi_y_start), isColor=False)

        if not (tracked_out.isOpened() and roi_out.isOpened() and mask_out.isOpened()):
            st.error("Error: Could not initialize video writers.")
            cap.release()
            os.unlink(video_path)
            st.stop()

        # Progress bar and status
        st.write("### Processing Video")
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract Region of Interest
            roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # Object Detection
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append([x, y, w, h])

            # Object Tracking
            boxes_ids = tracker.update(detections)
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Write frames to output videos
            tracked_out.write(frame)
            roi_out.write(roi)
            mask_out.write(mask)

            # Update progress
            processed_frames += 1
            progress = min(processed_frames / frame_count, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {processed_frames}/{frame_count} ({int(progress*100)}%)")

        # Release resources
        cap.release()
        tracked_out.release()
        roi_out.release()
        mask_out.release()

        # Verify output files
        for output_file in [tracked_output_file, roi_output_file, mask_output_file]:
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                st.error(f"Error: Output video {output_file} was not created successfully.")
                os.unlink(video_path)
                for f in [tracked_output_file, roi_output_file, mask_output_file]:
                    if os.path.exists(f):
                        os.unlink(f)
                st.stop()

        # Convert videos to base64
        tracked_b64 = video_to_base64(tracked_output_file)
        roi_b64 = video_to_base64(roi_output_file)
        mask_b64 = video_to_base64(mask_output_file)

        # Display results
        st.success("Processing complete!")
        st.write("### Tracked Video")
        html(f"""
            <div class="video-container">
                <video width="100%" controls autoplay muted>
                    <source src="data:video/mp4;base64,{tracked_b64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
        """, height=height//2)

        st.write("### ROI and Mask Videos")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ROI Video**")
            html(f"""
                <div class="video-container">
                    <video width="100%" controls autoplay muted>
                        <source src="data:video/mp4;base64,{roi_b64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            """, height=(roi_y_end - roi_y_start)//2)
        with col2:
            st.write("**Mask Video**")
            html(f"""
                <div class="video-container">
                    <video width="100%" controls autoplay muted>
                        <source src="data:video/mp4;base64,{mask_b64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            """, height=(roi_y_end - roi_y_start)//2)

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(tracked_output_file)
        os.unlink(roi_output_file)
        os.unlink(mask_output_file)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()
        for out in ['tracked_out', 'roi_out', 'mask_out']:
            if out in locals():
                locals()[out].release()
        for f in ['video_path', 'tracked_output_file', 'roi_output_file', 'mask_output_file']:
            if f in locals() and os.path.exists(locals()[f]):
                os.unlink(locals()[f])

else:
    st.info("Please upload a video file to start tracking. Supported formats: MP4, AVI, MOV.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and OpenCV | © 2025 Object Tracking App")