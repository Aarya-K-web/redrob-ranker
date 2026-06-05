import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

# --- Step 4: Sort by score, break ties by candidate_id ---
ranked = features_df.sort_values(
    ["final_score", "candidate_id"],
    ascending=[False, True]
).head(100).copy()
ranked["rank"] = range(1, 101)

# --- Step 5: Print top 10 sanity check ---
print("\n=== TOP 10 CANDIDATES ===")
for _, row in ranked.head(10).iterrows():
    print(f"Rank {int(row['rank']):2d} | {row['candidate_id']} | "
          f"Score: {row['final_score']:.4f} | "
          f"{str(row['current_title'])[:30]:30s} | "
          f"{row['country']}")

print(f"\n=== SCORE DISTRIBUTION ===")
print(f"Top score:    {ranked['final_score'].max():.4f}")
print(f"Median score: {ranked['final_score'].median():.4f}")
print(f"Bottom score: {ranked['final_score'].min():.4f}")
print(f"Honeypots eliminated: {(features_df['final_score'] == 0.0).sum()}")

# --- Step 6: Generate reasoning strings ---
def generate_reasoning(row):
    parts = []

    sem = row["semantic_score_norm"]
    if sem > 0.75:
        parts.append("Strong semantic alignment with JD requirements")
    elif sem > 0.50:
        parts.append("Good semantic match to role requirements")
    else:
        parts.append("Partial semantic match to role requirements")

    yoe = row["yoe"]
    if 5 <= yoe <= 9:
        parts.append(f"{yoe:.1f} years experience (ideal range)")
    elif yoe < 5:
        parts.append(f"{yoe:.1f} years experience (slightly junior)")
    else:
        parts.append(f"{yoe:.1f} years experience (senior)")

    if row["location_score"] == 1.0:
        parts.append("Based in India")
    elif row["location_score"] == 0.6:
        parts.append("Willing to relocate to India")
    else:
        parts.append("Location outside India, not willing to relocate")

    eng = row["engagement_score"]
    if eng > 0.7:
        parts.append("Highly engaged and available")
    elif eng > 0.4:
        parts.append("Moderately active on platform")
    else:
        parts.append("Low recent activity — may be hard to reach")

    if row["notice_days"] <= 30:
        parts.append(f"Short notice period ({int(row['notice_days'])} days)")
    elif row["notice_days"] <= 60:
        parts.append(f"Standard notice period ({int(row['notice_days'])} days)")
    else:
        parts.append(f"Long notice period ({int(row['notice_days'])} days)")

    if row["consulting_penalty"] < 1.0:
        parts.append("Note: entire career at IT services firms")

    return ". ".join(parts) + "."

ranked["reasoning"] = ranked.apply(generate_reasoning, axis=1)

# --- Step 7: Save submission CSV ---
submission = ranked[["candidate_id", "rank", "final_score", "reasoning"]].copy()
submission.columns = ["candidate_id", "rank", "score", "reasoning"]

output_file = "submission.csv"
submission.to_csv(output_file, index=False)
print(f"\nSaved {output_file} with {len(submission)} candidates")
print("\nFirst 3 rows of submission:")
print(submission.head(3).to_string())