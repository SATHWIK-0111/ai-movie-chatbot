import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load data
df = pd.read_csv("movies.csv")
embeddings = np.load("movie_embeddings.npy")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_movies(user_input):

    # Encode user input
    query_embedding = model.encode([user_input])

    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

    df["similarity"] = similarity_scores
    df_sorted = df.sort_values(by="similarity", ascending=False)

    top_movies = df_sorted.head(5)

    result = ""
    for _, row in top_movies.iterrows():
        result += f"🎬 {row['title']} ({row['genres']})\n\n"

    return result

iface = gr.Interface(
    fn=recommend_movies,
    inputs="text",
    outputs="text",
    title="AI Movie Recommendation Bot",
    description="Enter a movie or description and get recommendations."
)

iface.launch()