# hello.py

# Move st.set_page_config to the top before any imports
import streamlit as st
st.set_page_config(layout="wide")  # Must be the first Streamlit command

import pandas as pd
from modules.index import *

# dataset = pd.read_csv("../files/processed_data.csv")
dataset = pd.read_csv("../files/recipes_data.csv")

# Custom CSS to inject for styling the tabs
custom_css = """
    <style>
        /* Style the tab bar */
        .stTabs {
            justify-content: center; /* Center tabs */
        }
        /* Style each tab */
        .st-bb {
            flex-grow: 1; /* Each tab takes equal space */
            justify-content: center; /* Center text in each tab */
        }

        .stTextInput > div > div > input {
            background-color: white;  /* White background */
            color: black;  /* Black text color */
            width: 100%;  /* Full width */
            padding: 12px;  /* Padding for input */
            border: 1px solid #ddd;  /* Light border */
            border-radius: 24px;  /* Rounded corners */
            font-size: 16px;  /* Text size */
            outline: none;  /* Remove default outline */
            transition: box-shadow 0.2s ease-in-out;  /* Smooth focus effect */
        }
        .stTextInput > div > div > input:focus {
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);  /* Shadow on focus */
        }
        
    </style>
"""

