import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("../data/raw/movies.csv")

df.fillna({'genres':'','overview':''}, inplace=True)

df["combined_text"] = (
    df["genres"] + " " +
    df["overview"] + " " +
    df["title"]
)

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True
)

np.save("movie_embeddings.npy", embeddings)

print("Embeddings saved successfully.")