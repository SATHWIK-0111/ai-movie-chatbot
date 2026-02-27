from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from difflib import get_close_matches

# -------------------------------------------------
# Setup project root path
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.recommender import load_dataset, recommend, parse_input, is_details_query

# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI(title="AI Movie Recommendation API")

# -------------------------------------------------
# Enable CORS (for frontend connection)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Load dataset + embeddings ONCE at startup
# -------------------------------------------------
print("Loading recommender system...")

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "movies.csv")

df, embeddings, model = load_dataset(DATA_PATH)

print("Recommender loaded successfully!")

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------------------------
# Root endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "Movie Recommendation API is running 🚀"}

# -------------------------------------------------
# Chat endpoint
# -------------------------------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    user_input = request.message

    time_limit, top_k, genre, movies, lang = parse_input(user_input)

    # -------------------------------------------------
    # DETAILS QUERY
    # -------------------------------------------------
    if is_details_query(user_input) and movies:

        match = get_close_matches(
            movies[0].lower(),
            df['title'].str.lower(),
            n=1,
            cutoff=0.6
        )

        if match:
            row = df[df['title'].str.lower() == match[0]].iloc[0]

            return {
                "type": "details",
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "runtime": int(row["runtime"]),
                "overview": str(row["overview"])
            }

        return {
            "type": "details",
            "error": "Movie not found"
        }

    # -------------------------------------------------
    # RECOMMENDATION QUERY
    # -------------------------------------------------
    results = recommend(
        df,
        embeddings,
        model,
        user_input,
        time_limit,
        top_k,
        genre,
        movies,
        lang
    )

    # 🔥 Convert numpy types to Python native types
    clean_results = []

    for record in results.to_dict(orient="records"):
        clean_record = {
            "title": str(record["title"]),
            "genres": str(record["genres"]),
            "runtime": int(record["runtime"]),
            "similarity": float(record.get("similarity", 0)),
            "weighted_rating": float(record.get("weighted_rating", 0)),
            "runtime_comfort": float(record.get("runtime_comfort", 0)),
            "rating_confidence": float(record.get("rating_confidence", 0)),
            "score": float(record["score"]),
        }
        clean_results.append(clean_record)

    return {
        "type": "recommendation",
        "query": user_input,
        "recommendations": clean_results
    }