import json
from datetime import datetime, date

def build_text_blob(candidate):
    """
    Takes a candidate dict and returns a single string
    summarizing everything meaningful about them.
    This text will be fed into the embedding model.
    """
    p = candidate["profile"]
    parts = []

    # Basic identity
    parts.append(f"{p['current_title']} with {p['years_of_experience']} years of experience.")
    parts.append(f"Currently at {p['current_company']} in the {p['current_industry']} industry.")
    parts.append(f"Located in {p['location']}, {p['country']}.")

    # Their own summary
    if p.get("summary"):
        parts.append(p["summary"])

    # Career history - this is the most important part
    for job in candidate["career_history"]:
        parts.append(f"Worked as {job['title']} at {job['company']} "
                     f"({job['industry']}, {job['company_size']} employees) "
                     f"for {job['duration_months']} months.")
        if job.get("description"):
            parts.append(job["description"])

    # Skills with proficiency and duration
    skill_parts = []
    for skill in candidate["skills"]:
        duration = skill.get("duration_months", 0)
        skill_parts.append(f"{skill['name']} ({skill['proficiency']}, {duration} months)")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    # Education
    for edu in candidate.get("education", []):
        parts.append(f"Education: {edu['degree']} in {edu['field_of_study']} "
                     f"from {edu['institution']} ({edu.get('tier', 'unknown')} tier).")

    return " ".join(parts)


def extract_features(candidate):
    """
    Extracts structured numeric/boolean features from a candidate.
    These are used for rule-based scoring on top of the embedding score.
    """
    p = candidate["profile"]
    sig = candidate["redrob_signals"]

    # --- Location score ---
    india_cities = ["pune", "noida", "delhi", "mumbai", "bangalore", 
                    "bengaluru", "hyderabad", "chennai", "gurgaon", "india"]
    location_str = (p["location"] + " " + p["country"]).lower()
    is_india = any(city in location_str for city in india_cities)
    location_score = 1.0 if is_india else (0.6 if sig["willing_to_relocate"] else 0.1)

    # --- Experience score ---
    yoe = p["years_of_experience"]
    if 5 <= yoe <= 9:
        exp_score = 1.0       # perfect range
    elif 4 <= yoe < 5:
        exp_score = 0.8       # slightly junior but ok
    elif 9 < yoe <= 12:
        exp_score = 0.8       # slightly senior but ok
    elif yoe < 4:
        exp_score = 0.4       # too junior
    else:
        exp_score = 0.6       # very senior, deprioritize

    # --- Engagement / availability score ---
    last_active = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_inactive = (date.today() - last_active).days
    if days_inactive < 30:
        active_score = 1.0
    elif days_inactive < 90:
        active_score = 0.7
    elif days_inactive < 180:
        active_score = 0.3
    else:
        active_score = 0.0   # effectively gone

    response_score = sig["recruiter_response_rate"]  # already 0-1

    notice = sig["notice_period_days"]
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    open_to_work = 1.0 if sig["open_to_work_flag"] else 0.3

    engagement_score = (
        active_score * 0.4 +
        response_score * 0.2 +
        notice_score * 0.2 +
        open_to_work * 0.2
    )

    # --- Consulting-only penalty ---
    consulting_firms = ["tcs", "wipro", "infosys", "accenture", 
                        "cognizant", "capgemini", "hcl", "tech mahindra"]
    all_companies = [job["company"].lower() for job in candidate["career_history"]]
    entire_career_consulting = all(
        any(firm in company for firm in consulting_firms)
        for company in all_companies
    )
    consulting_penalty = 0.2 if entire_career_consulting else 1.0

    # --- Honeypot detection ---
    # Check 1: expert skills with 0 months experience
    expert_zero_months = sum(
        1 for s in candidate["skills"]
        if s["proficiency"] in ["expert", "advanced"] 
        and s.get("duration_months", 0) == 0
    )
    # Check 2: total career months vs stated years of experience
    total_career_months = sum(job["duration_months"] for job in candidate["career_history"])
    stated_months = yoe * 12
    experience_gap = stated_months - total_career_months
    
    is_honeypot = (expert_zero_months >= 4) or (experience_gap > 36)
    
    return {
        "candidate_id": candidate["candidate_id"],
        "text_blob": build_text_blob(candidate),
        "location_score": location_score,
        "exp_score": exp_score,
        "engagement_score": engagement_score,
        "consulting_penalty": consulting_penalty,
        "is_honeypot": is_honeypot,
        "yoe": yoe,
        "notice_days": notice,
        "country": p["country"],
        "current_title": p["current_title"],
    }


# Test it on the first 3 candidates
if __name__ == "__main__":
    with open("sample_candidates.json") as f:
        candidates = json.load(f)

    for c in candidates[:3]:
        features = extract_features(c)
        print(f"\n=== {features['candidate_id']} ===")
        print(f"Title: {features['current_title']}")
        print(f"Location score: {features['location_score']}")
        print(f"Experience score: {features['exp_score']} (YOE: {features['yoe']})")
        print(f"Engagement score: {features['engagement_score']:.2f}")
        print(f"Consulting penalty: {features['consulting_penalty']}")
        print(f"Is honeypot: {features['is_honeypot']}")
        print(f"Text blob (first 200 chars): {features['text_blob'][:200]}...")