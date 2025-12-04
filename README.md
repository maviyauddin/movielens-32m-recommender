<p align="center">
  <img src="https://raw.githubusercontent.com/maviyauddin/movielens-32m-recommender/main/MovieLens_Rec.png" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-MovieLens_32M-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-SVD%20|%20CF-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-orange?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/maviyauddin/movielens-32m-recommender/main/infograohic-style.png" width="100%">
</p>

## 📑 Table of Contents

- [Overview](#movielens-32m--end-to-end-recommendation-system)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [How to Run](#-how-to-run-the-notebooks)
- [Methods Used](#-methods-used)
- [Possible Extensions](#-possible-extensions)
- [Purpose](#-purpose-of-this-project)



# MovieLens 32M – End-to-End Recommendation System

This project is an end-to-end movie recommendation system built on the **MovieLens 32M** dataset.  
It showcases the full workflow a data scientist would follow:

- Data loading and exploratory data analysis (EDA)
- Baseline popularity-based recommendations
- Neighborhood-based collaborative filtering (Item–Item and User–User)
- Matrix factorization using SVD

The goal of this project is to demonstrate **practical recommender system skills** in a way that is clear, organized, and easy for recruiters or collaborators to review.

---

## 📂 Project Structure

```text
.
├── 01_data_loading_eda.ipynb
├── 02_item_item_and_user_user.ipynb
├── 03_matrix_factorization.ipynb
└── README.md
01_data_loading_eda.ipynb

Loads the MovieLens 32M data (movies, ratings, tags, links)

Performs structured EDA:

Dataset shapes, data types, and missing values

Rating distribution

User activity (ratings per user)

Movie popularity (ratings per movie)

Genre frequency analysis

Implements a baseline popularity-based recommender (Top-N globally popular movies)

02_item_item_and_user_user.ipynb

Builds a filtered user–item rating matrix (to stay within memory limits)

Implements Item–Item collaborative filtering:

Cosine similarity between movie vectors

recommend_similar_movies(title, top_n) → “More like this” style recommendations

Implements User–User collaborative filtering:

Normalized user rating profiles

User–user cosine similarity

recommend_for_user(user_id, top_n) → recommendations based on similar users

03_matrix_factorization.ipynb

Uses the Surprise library (SVD) for matrix factorization

Trains an SVD model on a sampled subset of MovieLens 32M (for efficient training)

Evaluates the model with RMSE

Implements recommend_for_user_svd(user_id, top_n) to generate recommendations based on predicted ratings.

🧾 Dataset
This project uses the MovieLens 32M dataset from GroupLens:

https://files.grouplens.org/datasets/movielens/ml-32m.zip

Note: The dataset is not included in this repository due to size.
To run the notebooks, download the dataset manually from the link above and place it in a folder structure similar to:

text
Copy code
ml32m/
└── ml-32m/
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    ├── links.csv
    ├── checksums.txt
    └── README.txt
In the notebooks, the data is loaded using paths like:

python
Copy code
movies_path = "ml32m/ml-32m/movies.csv"
ratings_path = "ml32m/ml-32m/ratings.csv"
🚀 How to Run the Notebooks
You can run these notebooks in either Google Colab or a local Python environment.

Option 1: Google Colab
Upload the notebooks (.ipynb files) to Colab or open directly from GitHub via Colab.

Upload or mount the MovieLens dataset so that the paths match:

python
Copy code
"ml32m/ml-32m/movies.csv"
"ml32m/ml-32m/ratings.csv"
Run cells in order.
For large-scale operations, the notebooks deliberately:

Use filtered subsets (top-N active users, top-N popular movies)

Use samples of ratings for SVD
to stay within typical Colab RAM and time limits.

Option 2: Local Environment
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/movielens-32m-recommender.git
cd movielens-32m-recommender
Create a virtual environment and install dependencies (example):

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-surprise
Place the MovieLens data in the ml32m/ml-32m/ folder.

Open the notebooks with Jupyter or VS Code and run them.

🧠 Methods Used
EDA & Baseline

Exploratory Data Analysis on users, movies, ratings, and genres

Popularity-based Top-N recommender

Collaborative Filtering

Item–Item collaborative filtering using cosine similarity

User–User collaborative filtering with normalized rating profiles

Matrix Factorization (SVD)

Surprise library SVD model

Train/test split and RMSE evaluation

Top-N recommendations via predicted ratings

🔮 Possible Extensions
Some possible future improvements:

Implement ranking metrics (Precision@K, Recall@K, NDCG)

Add implicit feedback signals (watch counts, clicks, etc.)

Experiment with neural recommendation models

Deploy a simple API or web demo for interactive recommendations

📌 Purpose of This Project
This project is designed as a portfolio-quality example to demonstrate:

Practical experience with real-world recommendation systems

Ability to handle larger datasets under resource constraints

Clear, organized notebooks with explanations and reproducible code

Feel free to explore, fork, or extend it.

