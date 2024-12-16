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

    with col2:
        generate_videos = st.button("Generate Dataset Videos")