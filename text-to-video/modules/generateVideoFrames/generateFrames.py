from diffusers import StableDiffusionPipeline
import os
import ast  # To safely evaluate string representation of lists

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cpu")  # Use CPU

# Function to generate images
def generate_image(prompt, output_path):
    print(f"[LOG] Generating image for prompt: {prompt}")
    try:
        image = pipe(prompt).images[0]
        image.save(output_path)
        print(f"[LOG] Image saved at: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate image for prompt: {prompt}. Error: {e}")

# Function to generate frames for a recipe
def generate_frames(recipe_title, ingredients, directions):
    print(f"[LOG] Processing recipe: {recipe_title}")
    recipe_folder = f"generated_frames/{recipe_title}"
    os.makedirs(recipe_folder, exist_ok=True)

    # Parse ingredients and directions from string to list
    if isinstance(ingredients, str):
        try:
            ingredients = ast.literal_eval(ingredients)
        except Exception as e:
            print(f"[ERROR] Failed to parse ingredients for recipe: {recipe_title}. Error: {e}")
            return

    if isinstance(directions, str):
        try:
            directions = ast.literal_eval(directions)
        except Exception as e:
            print(f"[ERROR] Failed to parse directions for recipe: {recipe_title}. Error: {e}")
            return

    # Generate images for ingredients
    for i, ingredient in enumerate(ingredients):
        ingredient_prompt = f"An image of {ingredient}"
        ingredient_path = os.path.join(recipe_folder, f"ingredient_{i+1}.png")
        print(f"[LOG] Creating image for ingredient: {ingredient}")
        generate_image(ingredient_prompt, ingredient_path)

    # Generate images for directions
    for i, step in enumerate(directions):
        direction_prompt = f"A visual representation of: {step}"
        direction_path = os.path.join(recipe_folder, f"direction_{i+1}.png")
        print(f"[LOG] Creating image for direction: {step}")
        generate_image(direction_prompt, direction_path)