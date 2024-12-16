import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def userPrompt():
    global is_processing
    st.subheader("Generate Frames and Video from User Prompt")

    # Define columns
    col1, col2 = st.columns([3, 1], gap="medium")  # Adjust column width and gap

    # Center-align the elements in the columns
    with col1:

    with col2:
        generate_videos = st.button("Generate Dataset Videos")