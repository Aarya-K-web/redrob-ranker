import json
import pandas as pd
from tqdm import tqdm
from features import extract_features

print("Re-extracting features for all 100K candidates...")
print("(Skipping embeddings — those are already saved)")

all_features = []

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Extracting"):
    candidate = json.loads(line.strip())
    features = extract_features(candidate)
    all_features.append({
        k: v for k, v in features.items() if k != "text_blob"
    })

features_df = pd.DataFrame(all_features)
features_df.to_csv("candidate_features.csv", index=False)
print(f"Done. Saved {len(features_df)} rows to candidate_features.csv")
print(f"Title penalties applied: {(features_df['title_penalty'] < 1.0).sum()} candidates penalized")