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