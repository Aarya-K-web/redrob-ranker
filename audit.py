import json
import pandas as pd

# Load submission
submission = pd.read_csv("submission.csv")

# Load top 100 candidate IDs
top_ids = set(submission["candidate_id"].tolist())

# Load their full profiles from the big file
print("Loading top 100 profiles from candidates.jsonl...")
top_candidates = {}
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line.strip())
        if c["candidate_id"] in top_ids:
            top_candidates[c["candidate_id"]] = c
        if len(top_candidates) == len(top_ids):
            break

print(f"Loaded {len(top_candidates)} profiles\n")

# Merge with submission scores
rows = []
for _, row in submission.iterrows():
    cid = row["candidate_id"]
    c = top_candidates.get(cid, {})
    if not c:
        continue
    p = c["profile"]
    sig = c["redrob_signals"]
    
    # Key retrieval skills
    skill_names = [s["name"].lower() for s in c["skills"]]
    career_text = " ".join([
        j.get("description", "") for j in c["career_history"]
    ]).lower()
    combined = " ".join(skill_names) + " " + career_text
    
    retrieval_hits = [s for s in [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "vector", "embedding", "retrieval",
        "semantic search", "ranking", "recommendation", "reranking"
    ] if s in combined]

    consulting_firms = [
        "tcs", "wipro", "infosys", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra"
    ]
    all_companies = [j["company"].lower() for j in c["career_history"]]
    entire_consulting = all(
        any(f in co for f in consulting_firms) for co in all_companies
    )

    rows.append({
        "rank": row["rank"],
        "candidate_id": cid,
        "score": row["score"],
        "title": p["current_title"],
        "company": p["current_company"],
        "industry": p["current_industry"],
        "yoe": p["years_of_experience"],
        "country": p["country"],
        "retrieval_skills": len(retrieval_hits),
        "retrieval_list": ", ".join(retrieval_hits[:5]),
        "entire_consulting": entire_consulting,
        "open_to_work": sig["open_to_work_flag"],
        "last_active": sig["last_active_date"],
        "notice_days": sig["notice_period_days"],
        "response_rate": sig["recruiter_response_rate"],
    })

df = pd.DataFrame(rows)

# --- Print audit sections ---
print("=" * 70)
print("TOP 10 — These should all be strong AI/ML engineers")
print("=" * 70)
for _, r in df[df["rank"] <= 10].iterrows():
    flag = "⚠️" if r["retrieval_skills"] == 0 else "✅"
    print(f"Rank {int(r['rank']):2d} {flag} | {r['title'][:35]:35s} | "
          f"YOE: {r['yoe']:.1f} | {r['country']}")
    print(f"         Retrieval skills ({r['retrieval_skills']}): "
          f"{r['retrieval_list'] or 'NONE'}")
    print(f"         Company: {r['company']} | "
          f"Consulting only: {r['entire_consulting']}")
    print()

print("=" * 70)
print("MIDDLE (ranks 45-55) — Should still be decent fits")
print("=" * 70)
for _, r in df[(df["rank"] >= 45) & (df["rank"] <= 55)].iterrows():
    flag = "⚠️" if r["retrieval_skills"] == 0 else "✅"
    print(f"Rank {int(r['rank']):2d} {flag} | {r['title'][:35]:35s} | "
          f"YOE: {r['yoe']:.1f} | {r['country']}")
    print(f"         Retrieval skills ({r['retrieval_skills']}): "
          f"{r['retrieval_list'] or 'NONE'}")
    print()

print("=" * 70)
print("BOTTOM 10 (ranks 91-100) — Weakest in your shortlist")
print("=" * 70)
for _, r in df[df["rank"] >= 91].iterrows():
    flag = "⚠️" if r["retrieval_skills"] == 0 else "✅"
    print(f"Rank {int(r['rank']):2d} {flag} | {r['title'][:35]:35s} | "
          f"YOE: {r['yoe']:.1f} | {r['country']}")
    print(f"         Retrieval skills ({r['retrieval_skills']}): "
          f"{r['retrieval_list'] or 'NONE'}")
    print()

print("=" * 70)
print("SUMMARY STATS")
print("=" * 70)
print(f"Candidates with 0 retrieval skills in top 100: "
      f"{(df['retrieval_skills'] == 0).sum()}")
print(f"Entire consulting career in top 100: "
      f"{df['entire_consulting'].sum()}")
print(f"Not open to work: {(~df['open_to_work']).sum()}")
print(f"Notice > 90 days: {(df['notice_days'] > 90).sum()}")
print(f"Outside India: {(df['country'] != 'India').sum()}")
print(f"Average retrieval skills per candidate: "
      f"{df['retrieval_skills'].mean():.1f}")