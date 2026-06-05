import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

print("Loading precomputed data...")
jd_embedding = np.load("jd_embedding.npy")
candidate_embeddings = np.load("candidate_embeddings.npy")
features_df = pd.read_csv("candidate_features.csv")

print(f"Loaded {len(features_df)} candidates")

# --- Step 1: Semantic similarity score ---
# Cosine similarity measures how "close" two vectors are in meaning.
# Score of 1.0 = identical meaning, 0.0 = completely unrelated, -1.0 = opposite.
# We reshape jd_embedding to (1, 384) so sklearn can compare it against all 100K.
print("Computing semantic similarity scores...")
semantic_scores = cosine_similarity(
    jd_embedding.reshape(1, -1),
    candidate_embeddings
)[0]  # [0] because output is shape (1, 100000) — we just want the array

features_df["semantic_score"] = semantic_scores

# --- Step 2: Normalize semantic scores to 0-1 range ---
# Raw cosine scores for text are usually between 0.1 and 0.4
# We normalize so they play nicely with our 0-1 rule scores
min_s = features_df["semantic_score"].min()
max_s = features_df["semantic_score"].max()
features_df["semantic_score_norm"] = (
    (features_df["semantic_score"] - min_s) / (max_s - min_s)
)

# --- Step 3: Calculate final hybrid score ---
# This is your core formula — combining all signals with weights
def calculate_final_score(row):
    # Immediately eliminate honeypots
    if row["is_honeypot"]:
        return 0.0

    # Weighted combination of all signals
    score = (
        row["semantic_score_norm"] * 0.40 +   # semantic match to JD
        row["exp_score"]           * 0.20 +   # right experience level
        row["engagement_score"]    * 0.20 +   # active and available
        row["location_score"]      * 0.20     # right location
    )

    # Apply consulting penalty (multiplier, not subtraction)
    score = score * row["consulting_penalty"]

    return round(score, 4)

print("Calculating final scores...")
features_df["final_score"] = features_df.apply(calculate_final_score, axis=1)

# --- Step 4: Sort and take top 100 ---
ranked = features_df.sort_values("final_score", ascending=False).head(100).copy()
ranked["rank"] = range(1, 101)

# Quick sanity check
print("\n=== TOP 10 CANDIDATES ===")
for _, row in ranked.head(10).iterrows():
    print(f"Rank {int(row['rank']):2d} | {row['candidate_id']} | "
          f"Score: {row['final_score']:.4f} | "
          f"{row['current_title'][:30]:30s} | "
          f"{row['country']}")

print(f"\n=== SCORE DISTRIBUTION ===")
print(f"Top score:    {ranked['final_score'].max():.4f}")
print(f"Median score: {ranked['final_score'].median():.4f}")
print(f"Bottom score: {ranked['final_score'].min():.4f}")
print(f"Honeypots eliminated: {(features_df['final_score'] == 0.0).sum()}")

# --- Step 5: Generate reasoning strings ---
def generate_reasoning(row):
    parts = []

    # Semantic match strength
    sem = row["semantic_score_norm"]
    if sem > 0.75:
        parts.append("Strong semantic alignment with JD requirements")
    elif sem > 0.50:
        parts.append("Good semantic match to role requirements")
    else:
        parts.append("Partial semantic match to role requirements")

    # Experience
    yoe = row["yoe"]
    if 5 <= yoe <= 9:
        parts.append(f"{yoe:.1f} years experience (ideal range)")
    elif yoe < 5:
        parts.append(f"{yoe:.1f} years experience (slightly junior)")
    else:
        parts.append(f"{yoe:.1f} years experience (senior)")

    # Location
    if row["location_score"] == 1.0:
        parts.append("Based in India")
    elif row["location_score"] == 0.6:
        parts.append("Willing to relocate to India")
    else:
        parts.append("Location outside India, not willing to relocate")

    # Availability
    eng = row["engagement_score"]
    if eng > 0.7:
        parts.append("Highly engaged and available")
    elif eng > 0.4:
        parts.append("Moderately active on platform")
    else:
        parts.append("Low recent activity — may be hard to reach")

    # Notice period
    if row["notice_days"] <= 30:
        parts.append(f"Short notice period ({int(row['notice_days'])} days)")
    elif row["notice_days"] <= 60:
        parts.append(f"Standard notice period ({int(row['notice_days'])} days)")
    else:
        parts.append(f"Long notice period ({int(row['notice_days'])} days)")

    # Consulting warning
    if row["consulting_penalty"] < 1.0:
        parts.append("Note: entire career at IT services firms")

    return ". ".join(parts) + "."

ranked["reasoning"] = ranked.apply(generate_reasoning, axis=1)

# --- Step 6: Save submission CSV ---
submission = ranked[["candidate_id", "rank", "final_score", "reasoning"]].copy()
submission.columns = ["candidate_id", "rank", "score", "reasoning"]

output_file = "submission.csv"
submission.to_csv(output_file, index=False)
print(f"\nSaved {output_file} with {len(submission)} candidates")
print("\nFirst 3 rows of submission:")
print(submission.head(3).to_string())