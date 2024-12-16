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
