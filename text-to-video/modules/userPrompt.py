import streamlit as st
import imageio
from diffusers import TextToVideoZeroPipeline
import datetime
import threading
import torch  # Optional, but ensure compatibility

# Lock and state to manage concurrent operations
lock = threading.Lock()
is_processing = False

# Video generation function
def generate_video(user_prompt):
    global is_processing
    try:
        # Load the TextToVideoZeroPipeline
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = TextToVideoZeroPipeline.from_pretrained(model_id)
        pipe = pipe.to("cpu")  # Use CPU for inference
        
        print(f"[LOG] Generating video for prompt: {user_prompt}")

        # Generate a unique output path using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"video_{timestamp}.mp4"
        
        # Generate frames
        result = pipe(prompt=user_prompt).images
        result = [(r * 255).astype("uint8") for r in result]

        # Save as video
        imageio.mimsave(output_path, result, fps=2)
        print(f"[LOG] Video saved at: {output_path}")
        st.success(f"Video saved successfully: {output_path}")
    except Exception as e:
        print(f"[ERROR] Video generation failed. Error: {e}")
        st.error(f"Error: {e}")
    finally:
        with lock:
            is_processing = False  # Reset processing flag after completion

# Streamlit UI for user input
def userPrompt():
    global is_processing
    st.subheader("Generate Video from User Prompt")

    # User input for the prompt
    user_prompt = st.text_area("Enter your prompt (e.g., 'A chef making cookies')")

    # Button to generate video
    generate_prompt_video = st.button("Generate Video")

    if generate_prompt_video and user_prompt.strip():
        with lock:
            if is_processing:
                st.warning("A process is already running. Please wait.")
                return  # Prevent multiple clicks while processing
            is_processing = True

        # Run the video generation in a separate thread
        threading.Thread(target=generate_video, args=(user_prompt,)).start()