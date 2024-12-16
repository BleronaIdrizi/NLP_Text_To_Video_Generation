import streamlit as st

def getAboutPage():
    st.header("About")
    st.subheader("Project Description")
    st.markdown("""
        This project was developed as part of the academic program at the University of Prishtina “Hasan Prishtina,” within the Faculty of Electrical and Computer Engineering for the course **Natural Language Processing**, under the guidance of Mërgim Hoti.
    """)
    st.markdown("""
        The goal of this project is to develop a model capable of interpreting natural language descriptions and generating corresponding short video clips. By advancing beyond static content creation, the project bridges the gap between text-based generation and multimedia generation. 
        To achieve this, we utilized a dataset showcasing the wonders of food composition. This dataset includes a wide variety of dishes, ranging from simple bread recipes to elaborate Swedish midsummer smorgasbords.
    """)

    st.subheader("Dataset Information")
    st.markdown("""
        The dataset used in this project is the **[Recipe Dataset (over 2M) Food](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)** from Kaggle. It contains recipes from around the world, featuring:
    """)
    st.markdown("""
        - **Columns:**
            - `title`: The title of the recipe.
            - `ingredients`: A list of ingredients and their amounts (unorganized format).
            - `directions`: Instructions for preparing the dish (unorganized format).
            - `link`: A URL linking to the original recipe source.
            - `source`: The source of the recipe (e.g., “Gathered” or “Recipes1M”).
            - `NER`: Organized ingredients without quantities, brands, or extra details.
            - `site`: The domain from which the recipe was gathered.
        - **Key Details:**
            - Total rows: 2.23 million.
            - Unique `title` values: 1,312,871.
            - Dataset size: ~2.31 GB.
        - **Usage:** The dataset is ideal for machine learning tasks related to natural language understanding and multimedia generation.
    """)

    st.subheader("Technologies Used")
    st.markdown("""
        The project was built using the following technologies and libraries:
        - **Python**: Core programming language.
        - **Machine Learning Frameworks**: Tools like PyTorch or TensorFlow (customize based on your project).
        - **Libraries for Data Processing and Analysis**:
            - Pandas for data manipulation.
            - Numpy for numerical operations.
            - Scikit-learn for preprocessing and clustering.
        - **Visualization**:
            - Matplotlib and Seaborn for data visualization.
            - Streamlit for building an interactive web app.
    """)

    st.markdown("""
        Additional resources and inspiration:
        - [Natural Language Processing with Python](https://realpython.com/natural-language-processing-python/)
        - [Streamlit Documentation](https://streamlit.io/)
        - [Kaggle Datasets](https://www.kaggle.com/datasets)
    """)

    st.subheader("Developed By")
    st.markdown("""
        This project was developed by:
        - **Blerona Idrizi** (blerona.idrizi@student.uni-pr.edu)
        - **Rina Shabani** (rina.shabani2@student.uni-pr.edu)
        - **Albiona Vukaj** (albiona.vukaj@student.uni-pr.edu)
    """, unsafe_allow_html=True)