import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from features import extract_features
from jd_text import JD_TEXT

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# This model is 80MB, fast on CPU, and very good at semantic similarity.
# It will auto-download on first run.

print("Embedding JD text...")
jd_embedding = model.encode(JD_TEXT, show_progress_bar=False)
np.save("jd_embedding.npy", jd_embedding)
print(f"JD embedding shape: {jd_embedding.shape}")

print("\nProcessing all candidates...")
all_features = []
all_text_blobs = []
candidate_ids = []

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total candidates: {len(lines)}")

for line in tqdm(lines, desc="Extracting features"):
    candidate = json.loads(line.strip())
    features = extract_features(candidate)
    all_features.append(features)
    all_text_blobs.append(features["text_blob"])
    candidate_ids.append(features["candidate_id"])

print("\nEmbedding all candidate text blobs...")
print("This will take 20-30 minutes. Go get a chai.")

# batch_size=64 means it processes 64 candidates at a time
# This is memory efficient and faster than one-by-one
candidate_embeddings = model.encode(
    all_text_blobs,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"\nCandidate embeddings shape: {candidate_embeddings.shape}")

# Save everything to disk
np.save("candidate_embeddings.npy", candidate_embeddings)

# Save features (without text blob to keep file small)
import pandas as pd
features_df = pd.DataFrame([
    {k: v for k, v in f.items() if k != "text_blob"}
    for f in all_features
])
features_df.to_csv("candidate_features.csv", index=False)

print("\nSaved:")
print("  candidate_embeddings.npy  — all 100K embeddings")
print("  candidate_features.csv    — all rule-based features")
print("  jd_embedding.npy          — JD embedding")
print("\nPrecompute complete! Now run rank.py")