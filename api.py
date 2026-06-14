import os
import json
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from features import get_flags

# --- App setup ---
app = FastAPI(
    title="Redrob Candidate Ranking Engine",
    description="Hybrid semantic + rule-based candidate ranking API",
    version="1.0.0"
)

# --- CORS Middleware Update ---
# Replaced wildcard with explicit origins to ensure absolute compatibility 
# between your Vercel frontend and Hugging Face backend.
origins = [
    "https://redrob-ranker.vercel.app",  # Your production Vercel frontend
    "http://localhost:3000",             # Common frontend local port (React/Next.js)
    "http://localhost:5173",             # Common frontend local port (Vite)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load everything once when server starts ---
print("Loading model and precomputed data...")
model = SentenceTransformer("all-MiniLM-L6-v2")
candidate_embeddings = np.load("candidate_embeddings.npy")
features_df = pd.read_csv("candidate_features.csv")

print("Loading candidate profiles...")
all_profiles = {}
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=100000):
        c = json.loads(line.strip())
        all_profiles[c["candidate_id"]] = c

print(f"Ready. Loaded {len(features_df):,} candidates.")

# --- Request/Response models ---
class RankRequest(BaseModel):
    job_description: str
    top_n: int = 20
    weight_semantic: float = 0.40
    weight_experience: float = 0.20
    weight_engagement: float = 0.20
    weight_location: float = 0.20

class CandidateResult(BaseModel):
    rank: int
    candidate_id: str
    score: float
    title: str
    company: str
    industry: str
    years_of_experience: float
    country: str
    notice_days: int
    open_to_work: bool
    retrieval_skills: list[str]
    reasoning: str
    green_flags: list[str]
    yellow_flags: list[str]
    red_flags: list[str]

class RankResponse(BaseModel):
    total_searched: int
    honeypots_removed: int
    results: list[CandidateResult]

# --- Helper functions ---
RETRIEVAL_SKILLS = [
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "vector", "embedding",
    "retrieval", "semantic search", "ranking", "recommendation",
    "reranking", "bm25", "hybrid search", "sentence-transformers"
]

CONSULTING_FIRMS = [
    "tcs", "wipro", "infosys", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra"
]

def get_retrieval_skills(candidate):
    skill_names = [s["name"].lower() for s in candidate["skills"]]
    career_text = " ".join([
        j.get("description", "") for j in candidate["career_history"]
    ]).lower()
    combined = " ".join(skill_names) + " " + career_text
    return [s for s in RETRIEVAL_SKILLS if s in combined]

def generate_reasoning(row, candidate):
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    parts = []

    parts.append(
        f"{p['current_title']} with {p['years_of_experience']:.1f} years "
        f"at {p['current_company']} ({p['current_industry']})"
    )

    retrieval = get_retrieval_skills(candidate)
    if retrieval:
        parts.append(f"Retrieval/ranking skills: {', '.join(retrieval[:5])}")
    else:
        parts.append("No explicit retrieval skills found")

    parts.append(
        f"Notice: {sig['notice_period_days']} days | "
        f"Last active: {sig['last_active_date']} | "
        f"Response rate: {sig['recruiter_response_rate']:.0%}"
    )

    concerns = []
    if row["location_score"] < 1.0 and not sig["willing_to_relocate"]:
        concerns.append(f"located in {p['country']}, not willing to relocate")
    if row["consulting_penalty"] < 1.0:
        concerns.append("entire career at IT services firms")
    if not retrieval:
        concerns.append("no vector search skills detected")
    if concerns:
        parts.append(f"Concerns: {'; '.join(concerns)}")

    return ". ".join(parts) + "."

# --- Routes ---
@app.get("/")
def root():
    return {
        "message": "Redrob Candidate Ranking Engine",
        "status": "running",
        "candidates_loaded": len(features_df),
        "endpoints": {
            "rank": "POST /rank",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "candidates": len(features_df)}

@app.post("/rank", response_model=RankResponse)
def rank_candidates(request: RankRequest):
    # 1. Embed the job description
    jd_embedding = model.encode(request.job_description)

    # 2. Compute cosine similarity
    scores = cosine_similarity(
        jd_embedding.reshape(1, -1),
        candidate_embeddings
    )[0]

    df = features_df.copy()
    df["semantic_score"] = scores

    # 3. Normalize
    min_s, max_s = df["semantic_score"].min(), df["semantic_score"].max()
    df["semantic_norm"] = (df["semantic_score"] - min_s) / (max_s - min_s) if max_s != min_s else 0.0

    # 4. Compute final score
    total_w = (
        request.weight_semantic +
        request.weight_experience +
        request.weight_engagement +
        request.weight_location
    )
    if total_w == 0:
        total_w = 1.0

    def score_row(row):
        if row["is_honeypot"]:
            return 0.0
        s = (
            row["semantic_norm"]      * request.weight_semantic +
            row["exp_score"]          * request.weight_experience +
            row["engagement_score"]   * request.weight_engagement +
            row["location_score"]     * request.weight_location
        ) / total_w
        return round(s * row["consulting_penalty"] * row["title_penalty"], 4)

    df["final_score"] = df.apply(score_row, axis=1)

    # 5. Sort and get top N
    ranked = df.sort_values(
        ["final_score", "candidate_id"],
        ascending=[False, True]
    ).head(request.top_n).copy()
    ranked["rank"] = range(1, len(ranked) + 1)

    honeypots_removed = int((df["final_score"] == 0.0).sum())

    # 6. Build response
    results = []
    for _, row in ranked.iterrows():
        cid = row["candidate_id"]
        candidate = all_profiles.get(cid, {})
        if not candidate:
            continue

        p = candidate["profile"]
        sig = candidate["redrob_signals"]
        retrieval = get_retrieval_skills(candidate)
        reasoning = generate_reasoning(row, candidate)
        green, yellow, red = get_flags(candidate)

        results.append(CandidateResult(
            rank=int(row["rank"]),
            candidate_id=cid,
            score=float(row["final_score"]),
            title=p["current_title"],
            company=p["current_company"],
            industry=p["current_industry"],
            years_of_experience=float(p["years_of_experience"]),
            country=p["country"],
            notice_days=int(sig["notice_period_days"]),
            open_to_work=bool(sig["open_to_work_flag"]),
            retrieval_skills=retrieval[:8],
            reasoning=reasoning,
            green_flags=green,
            yellow_flags=yellow,
            red_flags=red
        ))

    return RankResponse(
        total_searched=len(df),
        honeypots_removed=honeypots_removed,
        results=results
    )

# --- Direct Execution Setup for Hugging Face ---
if __name__ == "__main__":
    # Hugging Face sets the 'PORT' environment variable automatically to 7860.
    # If fallback to 8000 occurs, it's safe for your local environment run.
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)