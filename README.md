# Redrob Candidate Ranking Engine
### India Runs Hackathon — Track 1: AI & Datathon Arena

A hybrid AI system that ranks 100,000 candidates against a job description
using semantic embeddings + rule-based signals. Built for the Redrob
Senior AI Engineer role.

---

## How It Works

The system uses a two-stage pipeline:

**Stage 1 — Precompute (runs once, ~15 minutes)**
- Loads all 100,000 candidate profiles from `candidates.jsonl`
- Builds a rich text blob for each candidate combining their title,
  summary, career history, and skills
- Embeds all text blobs using `all-MiniLM-L6-v2` (sentence-transformers)
- Saves embeddings and rule-based features to disk

**Stage 2 — Rank (runs in ~60 seconds)**
- Loads precomputed embeddings from disk
- Computes cosine similarity between JD embedding and all 100K candidates
- Combines semantic score with rule-based signals into a hybrid score
- Outputs top 100 ranked candidates with explainable reasoning

---

## Scoring Formula
final_score = (
semantic_similarity  * 0.40 +   # cosine similarity to JD embedding
experience_score     * 0.20 +   # penalizes < 4 or > 12 years
engagement_score     * 0.20 +   # recency, response rate, notice period
location_score       * 0.20     # India-based or willing to relocate
) * consulting_penalty              # 0.2x if entire career at IT services firms

Honeypots are eliminated before scoring (score = 0.0).

---

## Honeypot Detection

Candidates are flagged as honeypots if:
- 4+ skills marked "expert" or "advanced" with 0 months of usage
- Stated years of experience exceeds total career history by 3+ years

38 honeypots were detected and eliminated from the 100K dataset.

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd redrob-ranker

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add data files (not included in repo due to size)
# Place candidates.jsonl in the project root

# 5. Run precompute (once)
python precompute.py

# 6. Run ranking
python rank.py

# 7. Validate submission
python validate_submission.py submission.csv
```

---

## Project Structure
redrob-ranker/
├── precompute.py          # Stage 1: embed all 100K candidates (run once)
├── rank.py                # Stage 2: score, rank, generate reasoning
├── features.py            # Feature extraction + honeypot detection
├── jd_text.py             # JD description text for embedding
├── explore.py             # Data exploration script
├── validate_submission.py # Provided by hackathon organizers
├── submission.csv         # Final ranked output (top 100)
├── requirements.txt       # Python dependencies
└── README.md

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Similarity | Cosine similarity via `scikit-learn` |
| Data processing | `pandas`, `numpy` |
| Rule-based scoring | Custom Python logic |
| Validation | Hackathon-provided `validate_submission.py` |

---

## Key Design Decisions

**Why all-MiniLM-L6-v2?**
Fast, lightweight (80MB), runs on CPU, and produces strong semantic
similarity scores for short-to-medium text. Fits within the 5-minute
runtime and 16GB RAM constraints.

**Why hybrid scoring instead of pure semantic?**
Semantic similarity alone doesn't capture availability, location, or
experience fit. A candidate in Canada who is not willing to relocate
could have perfect semantic match but be completely unreachable.
Rule-based signals handle these hard constraints cleanly.

**Why separate precompute and rank scripts?**
The hackathon requires ranking to complete in under 5 minutes on CPU
with no internet. Precomputing embeddings once and saving to disk
means the ranking step only loads files and does matrix math —
completing in under 60 seconds.

**Why penalize consulting-only careers?**
The job description explicitly states candidates whose entire career
is at TCS, Wipro, Infosys, Accenture, Cognizant, or Capgemini are
not a fit. The penalty is applied as a multiplier (0.2x) only when
every role in their career history is at such firms.