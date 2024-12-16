import streamlit as st
import os
import time
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont  # Replace with your text-to-image model
import ast  # For parsing string representations of lists
import requests

def generate_image_from_text(prompt, save_path):
    """
    Generates a realistic or artistic image based on the prompt using an AI model (e.g., Stable Diffusion or DALL-E).
    Replace the placeholder API logic with your text-to-image generation API.
    """
    st.write(f"Generating image with prompt: {prompt}")
    print(f"Generating image with prompt: {prompt}")
    
    # Placeholder: Use an external API for text-to-image generation
    api_url = "https://api.example.com/generate-image"  # Replace with the actual API endpoint
    headers = {"Authorization": "Bearer YOUR_API_KEY"}  # Replace with your API key
    data = {"prompt": prompt, "width": 512, "height": 512}  # Adjust width and height as needed
    
    response = requests.post(api_url, json=data, headers=headers)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)  # Save the image from the response
        st.write(f"Image saved at: {save_path}")
        print(f"Image saved at: {save_path}")
    else:
        st.error(f"Failed to generate image: {response.text}")
        print(f"Failed to generate image: {response.text}")

def generate_recipe_video(recipe_title, ingredients, directions):
    """
    Generate images for ingredients and directions, then combine them into a video.
    """
    # Prepare directories
    base_dir = "generated_frames"
    recipe_folder = os.path.join(base_dir, recipe_title.lower().replace(" ", "_"))
    if not os.path.exists(recipe_folder):
        os.makedirs(recipe_folder)
    
    st.write(f"Processing recipe: {recipe_title}")
    print(f"Processing recipe: {recipe_title}")
    
    # Generate images for ingredients
    frames = []
    for i, ingredient in enumerate(ingredients):
        frame_path = os.path.join(recipe_folder, f"ingredient_{i+1}.png")
        st.write(f"Creating image for ingredient: {ingredient}")
        print(f"Creating image for ingredient: {ingredient}")
        
        # Call the text-to-image generator for the ingredient
        generate_image_from_text(f"An image of {ingredient}", frame_path)
        frames.append(frame_path)
    
    # Generate images for directions
    for i, step in enumerate(directions):
        frame_path = os.path.join(recipe_folder, f"direction_{i+1}.png")
        st.write(f"Creating image for direction: {step}")
        print(f"Creating image for direction: {step}")
        
        # Call the text-to-image generator for the direction
        generate_image_from_text(f"A step-by-step guide for: {step}", frame_path)
        frames.append(frame_path)

    # Create a video from frames
    video_path = os.path.join(recipe_folder, f"{recipe_title.lower().replace(' ', '_')}.mp4")
    st.write(f"Combining frames into video: {video_path}")
    print(f"Combining frames into video: {video_path}")
    clip = ImageSequenceClip(frames, fps=1)
    clip.write_videofile(video_path, codec="libx264")
    
    return video_path


def generateVideos(df):
    st.subheader("Generating Recipe Videos")

    # Define the base directory for generated frames and videos
    base_dir = "generated_frames"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Placeholder for dynamic logs
    log_placeholder = st.empty()

    # Iterate through the dataset
    for _, row in df.iterrows():
        recipe_title = row['title']
        
        # Parse the ingredients and directions fields into lists
        ingredients = ast.literal_eval(row['ingredients'])  # Convert string to list
        directions = ast.literal_eval(row['directions'])  # Convert string to list
        
        st.write(f"Generating video for: {recipe_title}")
        print(f"Generating video for: {recipe_title}")
        video_path = generate_recipe_video(recipe_title, ingredients, directions)
        
        st.video(video_path)

    # Final success message
    log_placeholder.success("All videos generated successfully!")