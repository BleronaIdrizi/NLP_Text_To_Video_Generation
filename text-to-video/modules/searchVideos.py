import streamlit as st
import os

# Directory where videos are stored
video_directory = "../videos"

def searchVideos():
    # Function to list all video files in the directory
    def get_video_files(directory):
        return [f for f in os.listdir(directory) if f.endswith(".mp4")]

    # Function to display video names with underscores
    def display_video_names(videos):
        return [os.path.splitext(video)[0] for video in videos]

    # Load video files
    video_files = get_video_files(video_directory)
    video_names = display_video_names(video_files)

    # Streamlit app
    st.title("Generated Videos Viewer")
