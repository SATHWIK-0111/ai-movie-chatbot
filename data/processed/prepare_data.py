import pandas as pd
import os

# Load raw dataset
raw_path = "../raw/movies.csv"
movies = pd.read_csv(raw_path)

# Keep only required columns
movies = movies[
    [
        "title",
        "overview",
        "genres",
        "vote_average_norm",
        "popularity_norm"
    ]
]

# Drop null rows
movies = movies.dropna()

# Combine text fields
movies["combined_text"] = (
    movies["genres"] + " " +
    movies["overview"] + " " +
    movies["title"]
)

# Save cleaned dataset
processed_path = "movies_cleaned.csv"
movies.to_csv(processed_path, index=False)

print("Dataset cleaned and saved successfully.")