import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

print("Loading precomputed data...")
jd_embedding = np.load("jd_embedding.npy")
candidate_embeddings = np.load("candidate_embeddings.npy")
features_df = pd.read_csv("candidate_features.csv")

print(f"Loaded {len(features_df)} candidates")

# --- Step 1: Semantic similarity score ---
print("Computing semantic similarity scores...")
semantic_scores = cosine_similarity(
    jd_embedding.reshape(1, -1),
    candidate_embeddings
)[0]

features_df["semantic_score"] = semantic_scores

# --- Step 2: Normalize semantic scores to 0-1 range ---
min_s = features_df["semantic_score"].min()
max_s = features_df["semantic_score"].max()
features_df["semantic_score_norm"] = (
    (features_df["semantic_score"] - min_s) / (max_s - min_s)
)

# --- Step 3: Calculate final hybrid score ---
def calculate_final_score(row):
    if row["is_honeypot"]:
        return 0.0

    score = (
        row["semantic_score_norm"] * 0.40 +
        row["exp_score"]           * 0.20 +
        row["engagement_score"]    * 0.20 +
        row["location_score"]      * 0.20
    )

    score = score * row["consulting_penalty"]
    return round(score, 4)

print("Calculating final scores...")
features_df["final_score"] = features_df.apply(calculate_final_score, axis=1)

# --- Step 4: Sort and get top 100 ---
ranked = features_df.sort_values(
    ["final_score", "candidate_id"],
    ascending=[False, True]
).head(100).copy()
ranked["rank"] = range(1, 101)

# --- Step 5: Load original candidate data for top 100 ---
# We need real profile facts to write meaningful reasoning
print("Loading original profiles for top 100 candidates...")
top_ids = set(ranked["candidate_id"].tolist())
top_candidates = {}

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=100000, desc="Scanning candidates"):
        c = json.loads(line.strip())
        if c["candidate_id"] in top_ids:
            top_candidates[c["candidate_id"]] = c
        if len(top_candidates) == len(top_ids):
            break  # found all 100, stop early

print(f"Loaded {len(top_candidates)} candidate profiles")

# --- Step 6: Generate rich reasoning from real profile data ---
# Key AI/ML skills the JD cares about
RETRIEVAL_SKILLS = [
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "vector", "embedding", "retrieval", "semantic search",
    "sentence-transformers", "bge", "e5", "dense retrieval", "hybrid search",
    "bm25", "ranking", "recommendation", "reranking", "learning to rank"
]

LLM_SKILLS = [
    "llm", "fine-tuning", "lora", "qlora", "rag", "langchain", "openai",
    "huggingface", "transformers", "bert", "gpt", "nlp", "text classification"
]

CONSULTING_FIRMS = [
    "tcs", "wipro", "infosys", "accenture", "cognizant", 
    "capgemini", "hcl", "tech mahindra"
]

def get_relevant_skills(candidate, skill_list):
    """Find which skills from our list the candidate actually has."""
    candidate_skills = [s["name"].lower() for s in candidate["skills"]]
    # Also check career descriptions
    career_text = " ".join([
        job.get("description", "") for job in candidate["career_history"]
    ]).lower()
    
    found = []
    for skill in skill_list:
        if any(skill in cs for cs in candidate_skills) or skill in career_text:
            found.append(skill)
    return found

def get_best_companies(candidate):
    """Return notable non-consulting companies from career history."""
    good_companies = []
    for job in candidate["career_history"]:
        company = job["company"]
        is_consulting = any(
            firm in company.lower() for firm in CONSULTING_FIRMS
        )
        if not is_consulting:
            good_companies.append(f"{job['title']} at {job['company']}")
    return good_companies

def generate_rich_reasoning(row, candidate):
    """Generate reasoning using actual profile facts."""
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    parts = []

    # 1. Current role and experience
    parts.append(
        f"{p['current_title']} with {p['years_of_experience']:.1f} years of experience"
        f" at {p['current_company']} ({p['current_industry']})"
    )

    # 2. Retrieval/search skills found
    retrieval_found = get_relevant_skills(candidate, RETRIEVAL_SKILLS)
    if retrieval_found:
        parts.append(
            f"Relevant retrieval/ranking skills: {', '.join(retrieval_found[:5])}"
        )
    else:
        parts.append("No explicit retrieval or vector search skills found")

    # 3. LLM/NLP skills
    llm_found = get_relevant_skills(candidate, LLM_SKILLS)
    if llm_found:
        parts.append(f"LLM/NLP experience: {', '.join(llm_found[:4])}")

    # 4. Best career highlights (non-consulting)
    good_roles = get_best_companies(candidate)
    if good_roles:
        parts.append(
            f"Product company experience: {'; '.join(good_roles[:2])}"
        )

    # 5. Availability facts
    parts.append(
        f"Notice period: {sig['notice_period_days']} days"
        f" | Last active: {sig['last_active_date']}"
        f" | Response rate: {sig['recruiter_response_rate']:.0%}"
    )

    # 6. Honest concerns
    concerns = []
    if row["location_score"] < 1.0 and not sig["willing_to_relocate"]:
        concerns.append(f"located in {p['country']}, not willing to relocate")
    if row["consulting_penalty"] < 1.0:
        concerns.append("entire career at IT services firms")
    if not retrieval_found:
        concerns.append("no vector search or retrieval skills detected")
    if sig["recruiter_response_rate"] < 0.2:
        concerns.append(f"low recruiter response rate ({sig['recruiter_response_rate']:.0%})")
    if p["years_of_experience"] < 4:
        concerns.append("below minimum experience requirement")

    if concerns:
        parts.append(f"Concerns: {'; '.join(concerns)}")

    return ". ".join(parts) + "."

# Apply rich reasoning to all top 100
print("Generating rich reasoning for top 100...")
reasonings = []
for _, row in ranked.iterrows():
    cid = row["candidate_id"]
    if cid in top_candidates:
        reasoning = generate_rich_reasoning(row, top_candidates[cid])
    else:
        reasoning = "Profile data unavailable."
    reasonings.append(reasoning)

ranked["reasoning"] = reasonings

# --- Step 7: Print top 10 sanity check ---
print("\n=== TOP 10 CANDIDATES ===")
for _, row in ranked.head(10).iterrows():
    print(f"\nRank {int(row['rank'])} | {row['candidate_id']} | Score: {row['final_score']:.4f}")
    print(f"  {row['current_title']} | {row['country']}")
    print(f"  {row['reasoning'][:200]}...")

print(f"\n=== SCORE DISTRIBUTION ===")
print(f"Top score:    {ranked['final_score'].max():.4f}")
print(f"Median score: {ranked['final_score'].median():.4f}")
print(f"Bottom score: {ranked['final_score'].min():.4f}")
print(f"Honeypots eliminated: {(features_df['final_score'] == 0.0).sum()}")

# --- Step 8: Save submission CSV ---
submission = ranked[["candidate_id", "rank", "final_score", "reasoning"]].copy()
submission.columns = ["candidate_id", "rank", "score", "reasoning"]

output_file = "submission.csv"
submission.to_csv(output_file, index=False)
print(f"\nSaved {output_file} with {len(submission)} candidates")