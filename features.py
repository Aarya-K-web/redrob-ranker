import json
from datetime import datetime, date

# ── Consulting firms ──────────────────────────────────────────────
CONSULTING_FIRMS = [
    "tcs", "wipro", "infosys", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
    "niit", "mastech", "syntel", "patni", "igate"
]

# ── Wrong titles ──────────────────────────────────────────────────
UNRELATED_TITLES = [
    "hr manager", "human resources", "accountant", "civil engineer",
    "mechanical engineer", "marketing manager", "sales manager",
    "operations manager", "customer support", "business analyst",
    "product manager", "project manager", "finance manager",
    "teacher", "content writer", "graphic designer", "recruiter",
    "data entry", "office manager", "administrative", "lawyer",
    "doctor", "architect", "supply chain", "procurement"
]

# ── Semantic skill clusters (for expansion) ───────────────────────
RETRIEVAL_SKILLS = [
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "vector", "embedding",
    "retrieval", "semantic search", "ranking", "recommendation",
    "reranking", "bm25", "hybrid search", "sentence-transformers",
    "dense retrieval", "hnsw", "ann", "approximate nearest neighbor",
    "learning to rank", "information retrieval", "neural search"
]

LLM_SKILLS = [
    "llm", "fine-tuning", "lora", "qlora", "rag", "langchain",
    "openai", "huggingface", "transformers", "bert", "gpt",
    "nlp", "text classification", "named entity recognition",
    "question answering", "text generation", "prompt engineering"
]

MLOPS_SKILLS = [
    "mlflow", "kubeflow", "airflow", "model serving", "triton",
    "torchserve", "bentoml", "ray serve", "docker", "kubernetes",
    "ci/cd", "model monitoring", "feature store", "data pipeline"
]

PRODUCT_COMPANY_SIGNALS = [
    "google", "microsoft", "amazon", "apple", "meta", "flipkart",
    "swiggy", "zomato", "cred", "meesho", "razorpay", "phonepe",
    "paytm", "byju", "unacademy", "ola", "uber", "netflix",
    "spotify", "atlassian", "freshworks", "zoho", "cleartax",
    "sharechat", "dailyhunt", "dream11", "groww", "zerodha",
    "nykaa", "myntra", "bigbasket", "dunzo", "urban company",
    "yellow.ai", "rephrase.ai", "niramai", "mad street den",
    "aganitha", "haptik", "observe.ai", "sarvam", "krutrim"
]


def build_text_blob(candidate):
    """Builds rich text blob for embedding."""
    p = candidate["profile"]
    parts = []

    parts.append(
        f"{p['current_title']} with {p['years_of_experience']} years of experience."
    )
    parts.append(
        f"Currently at {p['current_company']} in the {p['current_industry']} industry."
    )
    parts.append(f"Located in {p['location']}, {p['country']}.")

    if p.get("summary"):
        parts.append(p["summary"])

    for job in candidate["career_history"]:
        parts.append(
            f"Worked as {job['title']} at {job['company']} "
            f"({job['industry']}, {job['company_size']} employees) "
            f"for {job['duration_months']} months."
        )
        if job.get("description"):
            parts.append(job["description"])

    skill_parts = []
    for skill in candidate["skills"]:
        duration = skill.get("duration_months", 0)
        skill_parts.append(
            f"{skill['name']} ({skill['proficiency']}, {duration} months)"
        )
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    for edu in candidate.get("education", []):
        parts.append(
            f"Education: {edu['degree']} in {edu['field_of_study']} "
            f"from {edu['institution']} ({edu.get('tier', 'unknown')} tier)."
        )

    return " ".join(parts)


def get_title_penalty(title):
    title_lower = title.lower()
    for bad in UNRELATED_TITLES:
        if bad in title_lower:
            return 0.1
    return 1.0


def score_location(p, sig):
    india_cities = [
        "pune", "noida", "delhi", "mumbai", "bangalore", "bengaluru",
        "hyderabad", "chennai", "gurgaon", "india", "kolkata", "ahmedabad",
        "jaipur", "lucknow", "kochi", "chandigarh"
    ]
    location_str = (p["location"] + " " + p["country"]).lower()
    is_india = any(city in location_str for city in india_cities)
    if is_india:
        return 1.0
    elif sig["willing_to_relocate"]:
        return 0.5
    else:
        return 0.1


def score_experience(p, candidate):
    """
    Scores experience with:
    - YOE range fit
    - Career trajectory (upward mobility)
    - Product company signal
    - Domain continuity
    - Recency bias (last 3 years weighted 2x)
    """
    yoe = p["years_of_experience"]

    # YOE fit score
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5:
        yoe_score = 0.8
    elif 9 < yoe <= 12:
        yoe_score = 0.75
    elif yoe < 4:
        yoe_score = 0.4
    else:
        yoe_score = 0.6

    # Product company signal
    product_score = 0.0
    consulting_count = 0
    total_roles = len(candidate["career_history"])

    for job in candidate["career_history"]:
        company_lower = job["company"].lower()
        is_product = any(p in company_lower for p in PRODUCT_COMPANY_SIGNALS)
        is_consulting = any(f in company_lower for f in CONSULTING_FIRMS)

        if is_product:
            product_score = min(1.0, product_score + 0.4)
        elif not is_consulting:
            product_score = min(1.0, product_score + 0.15)

        if is_consulting:
            consulting_count += 1

    # Career trajectory — check if titles are getting more senior
    titles = [job["title"].lower() for job in candidate["career_history"]]
    senior_signals = ["senior", "lead", "staff", "principal", "head", "director", "vp"]
    recent_senior = any(s in titles[0] for s in senior_signals) if titles else False
    trajectory_bonus = 0.1 if recent_senior else 0.0

    # Domain continuity — are they staying in AI/ML/tech?
    ai_domains = ["ai", "ml", "machine learning", "data science", "nlp",
                  "deep learning", "research", "engineering", "software"]
    ai_role_count = sum(
        1 for t in titles
        if any(d in t for d in ai_domains)
    )
    domain_score = min(1.0, ai_role_count / max(total_roles, 1))

    # Combine
    exp_score = (
        yoe_score * 0.40 +
        product_score * 0.35 +
        domain_score * 0.15 +
        trajectory_bonus * 0.10
    )

    return min(1.0, exp_score)


def score_engagement(sig):
    """
    Scores engagement with recency, response rate,
    notice period, and open-to-work signals.
    """
    last_active = datetime.strptime(
        sig["last_active_date"], "%Y-%m-%d"
    ).date()
    days_inactive = (date.today() - last_active).days

    if days_inactive < 14:
        active_score = 1.0
    elif days_inactive < 30:
        active_score = 0.9
    elif days_inactive < 60:
        active_score = 0.7
    elif days_inactive < 90:
        active_score = 0.5
    elif days_inactive < 180:
        active_score = 0.3
    else:
        active_score = 0.0

    response_score = sig["recruiter_response_rate"]

    notice = sig["notice_period_days"]
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    open_score = 1.0 if sig["open_to_work_flag"] else 0.3

    github_score = min(1.0, sig.get("github_activity_score", 0) / 100)

    return (
        active_score    * 0.35 +
        response_score  * 0.20 +
        notice_score    * 0.20 +
        open_score      * 0.15 +
        github_score    * 0.10
    )


def detect_honeypot(candidate):
    """
    Enhanced honeypot detection — flags on 2+ signals.
    """
    flags = 0
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    skills = candidate["skills"]

    # Signal 1: Expert skills with 0 months
    expert_zero = sum(
        1 for s in skills
        if s["proficiency"] in ["expert", "advanced"]
        and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 4:
        flags += 1

    # Signal 2: YOE vs actual career months gap
    total_career_months = sum(
        job["duration_months"] for job in candidate["career_history"]
    )
    stated_months = p["years_of_experience"] * 12
    if stated_months - total_career_months > 36:
        flags += 1

    # Signal 3: Too many skills (50+) with no evidence
    if len(skills) > 50:
        flags += 1

    # Signal 4: All career entries have identical duration
    durations = [job["duration_months"] for job in candidate["career_history"]]
    if len(durations) > 2 and len(set(durations)) == 1:
        flags += 1

    # Signal 5: Last active more than 2 years ago
    try:
        last_active = datetime.strptime(
            sig["last_active_date"], "%Y-%m-%d"
        ).date()
        if (date.today() - last_active).days > 730:
            flags += 1
    except Exception:
        flags += 1

    # Signal 6: Zero engagement but 100% profile signals
    if (sig["recruiter_response_rate"] == 0 and
            sig["open_to_work_flag"] and
            len(skills) > 20):
        flags += 1

    return flags >= 2


def get_flags(candidate):
    """
    Returns green, yellow, red flags for explainability.
    """
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    skills = candidate["skills"]
    career = candidate["career_history"]

    skill_names = [s["name"].lower() for s in skills]
    career_text = " ".join(
        j.get("description", "") for j in career
    ).lower()
    combined = " ".join(skill_names) + " " + career_text

    retrieval_found = [s for s in RETRIEVAL_SKILLS if s in combined]
    llm_found = [s for s in LLM_SKILLS if s in combined]
    mlops_found = [s for s in MLOPS_SKILLS if s in combined]

    company_names = [j["company"].lower() for j in career]
    at_product = any(
        any(p in c for p in PRODUCT_COMPANY_SIGNALS)
        for c in company_names
    )
    all_consulting = all(
        any(f in c for f in CONSULTING_FIRMS)
        for c in company_names
    )

    green = []
    yellow = []
    red = []

    # Green flags
    if retrieval_found:
        green.append(
            f"Retrieval/ranking skills: {', '.join(retrieval_found[:4])}"
        )
    if at_product:
        green.append("Product company background")
    if sig["open_to_work_flag"]:
        green.append("Actively open to work")
    if sig["notice_period_days"] <= 30:
        green.append(f"Quick joiner ({sig['notice_period_days']} day notice)")
    if sig.get("github_activity_score", 0) > 70:
        green.append("Strong GitHub activity")
    if llm_found:
        green.append(f"LLM/NLP skills: {', '.join(llm_found[:3])}")
    if mlops_found:
        green.append(f"MLOps experience: {', '.join(mlops_found[:3])}")

    # Yellow flags
    if not retrieval_found:
        yellow.append("No explicit vector search or retrieval skills found")
    if sig["notice_period_days"] > 60:
        yellow.append(f"Long notice period ({sig['notice_period_days']} days)")
    if sig["recruiter_response_rate"] < 0.3:
        yellow.append(
            f"Low recruiter response rate ({sig['recruiter_response_rate']:.0%})"
        )
    if p["years_of_experience"] < 5:
        yellow.append(
            f"Slightly below experience target ({p['years_of_experience']:.1f} yrs)"
        )
    if p["years_of_experience"] > 10:
        yellow.append(
            f"Over-experienced for role ({p['years_of_experience']:.1f} yrs)"
        )

    # Red flags
    if all_consulting:
        red.append("Entire career at IT services firms — JD explicitly excludes this")
    if p["country"] != "India" and not sig["willing_to_relocate"]:
        red.append(
            f"Located in {p['country']}, not willing to relocate to India"
        )

    return green, yellow, red


def extract_features(candidate):
    p = candidate["profile"]
    sig = candidate["redrob_signals"]

    location_score = score_location(p, sig)
    exp_score = score_experience(p, candidate)
    engagement_score = score_engagement(sig)

    company_names = [j["company"].lower() for j in candidate["career_history"]]
    entire_consulting = all(
        any(f in c for f in CONSULTING_FIRMS) for c in company_names
    )
    consulting_penalty = 0.15 if entire_consulting else 1.0

    title_penalty = get_title_penalty(p["current_title"])
    is_honeypot = detect_honeypot(candidate)

    return {
        "candidate_id": candidate["candidate_id"],
        "text_blob": build_text_blob(candidate),
        "location_score": location_score,
        "exp_score": exp_score,
        "engagement_score": engagement_score,
        "consulting_penalty": consulting_penalty,
        "title_penalty": title_penalty,
        "is_honeypot": is_honeypot,
        "yoe": p["years_of_experience"],
        "notice_days": sig["notice_period_days"],
        "country": p["country"],
        "current_title": p["current_title"],
    }


# Test
if __name__ == "__main__":
    with open("sample_candidates.json") as f:
        candidates = json.load(f)

    for c in candidates[:3]:
        features = extract_features(c)
        green, yellow, red = get_flags(c)
        print(f"\n=== {features['candidate_id']} — {features['current_title']} ===")
        print(f"Location: {features['location_score']:.2f} | "
              f"Exp: {features['exp_score']:.2f} | "
              f"Engage: {features['engagement_score']:.2f}")
        print(f"Consulting penalty: {features['consulting_penalty']} | "
              f"Honeypot: {features['is_honeypot']}")
        print(f"✅ Green: {green}")
        print(f"⚠️  Yellow: {yellow}")
        print(f"🔴 Red: {red}")