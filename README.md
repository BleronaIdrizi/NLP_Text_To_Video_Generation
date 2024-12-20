# Text-to-Video Generation

This project was developed as part of the academic program at the University of Prishtina “Hasan Prishtina,” within the Faculty of Electrical and Computer Engineering for the course Natural Language Processing, under the guidance of Mërgim Hoti.

The objective of this project is to develop a model that can interpret natural language descriptions and generate corresponding short video clips. The project aims to bridge the gap between text-based generation and multimedia generation by advancing beyond static content creation. To achieve this goal, a dataset showcasing the wonders of food composition was utilized. This dataset includes a wide variety of dishes, ranging from simple bread recipes to elaborate Swedish midsummer smorgasbords. It is available on the **[Recipe Dataset (over 2M) Food](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)** through the Kaggle platform.

## Dataset details

The dataset used in this project contains recipes from around the world, ranging from simple dishes like bread to complex meals like a Swedish midsummer smorgasbord. This dataset is available on the Kaggle platform and can be accessed through the following link:

**[Recipe Dataset (over 2M) Food](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)**

Key features of the dataset:
- **Dataset columns:** The dataset contains the following columns:
    - title: The title of the recipe.
    - ingredients: A list of ingredients and their amounts (unorganized format).
    - directions: Instructions for preparing the dish (unorganized format).
    - link: A URL linking to the original recipe source.
    - source: The source of the recipe (e.g., “Gathered” or “Recipes1M”).
    - NER: Organized ingredients without quantities, brands, or extra details.
    - site: The domain from which the recipe was gathered.
- **Number of rows and unique values:** 
    - Total rows: 2.23 million.
    - Unique values in key columns:
        - title: 1,312,871 unique values.
	    - ingredients: 2,226,362 unique values.
	    - directions: 2,211,644 unique values.
	    - NER: 200,476 unique values.
- **Dataset size:** The dataset has a size of approximately 2.31 GB.
- **Attribute types:**
    1. Categorical (Qualitative)
        - Nominale: 'title', 'ingredients', 'directions', 'link', 'source', 'NER', 'site'
    2. Textual
        - Columns such as ingredients and directions contain long textual data, which can be further analyzed or structured as needed.
- **Special Features:**
    - Some columns, like ingredients and NER, contain data in array format. To work with these arrays, you can use the df.apply() method and the json module.
    - Duplicate values have been removed in the NER column, and all values have been standardized to lowercase for consistency.
- **Source:** The dataset has been compiled from various online sources and published on Kaggle for use in machine learning projects and other applications.


## Steps to fix ffmpeg path issue

- The FileNotFoundError was caused because moviepy was defaulting to /usr/local/bin/ffmpeg, even though FFmpeg was installed at /opt/homebrew/bin/ffmpeg.
- This happened because some parts of moviepy rely on hardcoded paths and did not respect the environment variable IMAGEIO_FFMPEG_EXE.

### Steps to fix the issue:
1.  Verified the Correct FFmpeg Path:
- which ffmpeg
- This confirmed the correct FFmpeg binary path was /opt/homebrew/bin/ffmpeg
2. Verified FFmpeg’s availability and codecs:
ffmpeg -version
ffmpeg -codecs | grep libx264

## Libraries Used
This project leverages a range of Python libraries to handle data processing, natural language interpretation, video generation, and more. Below is the categorization of the libraries used:

### General-Purpose Libraries
- **pandas:** For data manipulation and analysis, including handling the recipe dataset.
- **numpy:** For numerical operations and handling multidimensional arrays.
- **seaborn:** For creating visualizations to analyze and understand data trends.

### Text-to-Video and Natural Language Processing (NLP) Libraries
- **transformers:** For implementing state-of-the-art NLP models to process and interpret text descriptions.
- **diffusers:** To utilize the Stable Diffusion model for generating high-quality images from text, which are later used for video creation.

### Video and Image Handling Libraries
- **moviepy:** For video manipulation, including combining image sequences into video clips and adding audio - overlays.
- **cv2 (OpenCV):** For advanced image processing tasks, such as resizing and applying filters.

### Text-to-Speech Libraries
- **gTTS (Google Text-to-Speech):** For converting text descriptions into speech, which can be embedded into video outputs.

### Miscellaneous Libraries
- **os:** For managing file paths and system-level operations.
- **json:** For parsing and manipulating JSON data structures, such as the ingredients and NER columns in the dataset.

### Additional Tools
- **ffmpeg:** Required for multimedia processing, including video encoding and audio integration.