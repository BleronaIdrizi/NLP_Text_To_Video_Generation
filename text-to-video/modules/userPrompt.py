import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import imageio
from diffusers import TextToVideoZeroPipeline
import datetime
import threading  # To manage concurrent operations

colors = sns.color_palette('pastel')  # Palette for visuals

# Define a global flag to track if a process is running
is_processing = False
lock = threading.Lock()  # Ensure thread-safe access to the flag


def userPrompt():
    global is_processing
    st.subheader("Generate Frames and Video from User Prompt")

    # Define columns
    col1, col2 = st.columns([3, 1], gap="medium")  # Adjust column width and gap

    # Center-align the elements in the columns
    with col1:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
                <label style="font-size: 16px; font-weight: bold;">Enter your prompt (e.g., 'A chef making cookies')</label>
                <textarea rows="5" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 8px;" id="user_prompt"></textarea>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        generate_videos = st.button("Generate Dataset Videos")

    # Process the input and button click
    user_prompt = st.session_state.get("user_prompt", "")

    if user_prompt.strip() and generate_videos:
        with lock:
            if is_processing:
                st.warning("A process is already running. Please wait until it completes.")
                return  # Exit if a process is already running
            is_processing = True  # Set the flag to indicate processing has started

        # Run the video generation in a separate thread
        threading.Thread(target=generate_video, args=(user_prompt,)).start()