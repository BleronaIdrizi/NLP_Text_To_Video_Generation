import streamlit as st
import os
from natsort import natsorted
import cv2

# Initialize session state at the very top
if 'generate_clicked' not in st.session_state:
    st.session_state.generate_clicked = False

# Paths
frame_directory = "../data_preparation/generated_frames"
video_directory = "../videos"

os.makedirs(video_directory, exist_ok=True)

# Video settings
FPS = 2  # Frames per second

def create_video_from_frames(recipe_name, image_paths, output_path):
    if not image_paths:
        print(f"[ERROR] No frames found for {recipe_name}")
        return
    
    # Sort images naturally (step_1, step_2, etc.)
    image_paths = natsorted(image_paths)
    
    # Read the first image to get dimensions
    frame = cv2.imread(image_paths[0])
    if frame is None:
        print(f"[ERROR] Failed to read the first frame: {image_paths[0]}")
        return
    
    h, w, _ = frame.shape
    print(f"[INFO] Frame size: {w}x{h}")
    
    # Use 'avc1' (H.264) codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for .mp4 files
    video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))
    
    frame_count = 0

    # Write each frame to the video
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"[WARNING] Skipping unreadable frame: {image_path}")
            continue
        
        # Resize if frame dimensions do not match
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
            print(f"[INFO] Resized frame: {image_path}")
        
        video_writer.write(frame)
        frame_count += 1
        print(f"[INFO] Added frame: {image_path}")
    
    video_writer.release()

    if frame_count == 0:
        print(f"[ERROR] No frames were written for {recipe_name}. Deleting empty video.")
        os.remove(output_path)
    else:
        print(f"[SUCCESS] Video created: {output_path}")

def collect_frames_and_generate_videos(base_dir):
    for root, dirs, files in os.walk(base_dir):
        image_paths = []
        recipe_name = os.path.basename(root)
        
        # Collect only .png files in the current directory
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
        
        if image_paths:
            # Generate video for each recipe/folder
            output_path = os.path.join(video_directory, f"{recipe_name}.mp4")
            create_video_from_frames(recipe_name, image_paths, output_path)

def searchVideos():
    # Function to list all video files in the directory
    def get_video_files(directory):
        return [f for f in os.listdir(directory) if f.endswith(".mp4")]

    # Function to display video names with underscores
    def display_video_names(videos):
        return [os.path.splitext(video)[0] for video in videos]

    # Streamlit app
    st.title("Generated Videos Viewer")
    
    # Generate videos button (sets session state)
    if st.button("Generate all frame videos"):
        st.session_state.generate_clicked = True

    # Check if button was clicked
    if st.session_state.generate_clicked:
        st.info("Generating videos... Please wait.")
        collect_frames_and_generate_videos(frame_directory)
        st.success("All videos have been generated!")
        st.session_state.generate_clicked = False

    # Load video files
    video_files = get_video_files(video_directory)
    video_names = display_video_names(video_files)

    # Search bar
    search_query = st.text_input("Search for a video by name:", "")

    # Show "View All Generated Videos" only if there's no search query
    if not search_query.strip():
        with st.expander("View All Generated Videos"):
            for name in video_names:
                st.write(name.replace("_", " "))  # Display with underscores as spaces

    # Filtered video list based on search query
    filtered_videos = [name for name in video_names if search_query.lower() in name.lower()]

    if search_query.strip():
        st.subheader("Search Results:")
        if filtered_videos:
            for name in filtered_videos:
                # Button for each video name
                if st.button(f"See video: {name}.mp4"):
                    # Display the video when clicked
                    video_path = os.path.join(video_directory, f"{name}.mp4")
                    st.video(video_path)
        else:
            st.write("No videos found for your search.")