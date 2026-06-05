import json

# Load the first 5 candidates from the sample file
with open("sample_candidates.json") as f:
    candidates = json.load(f)

# Look at the first candidate in detail
c = candidates[0]

print("=== BASIC INFO ===")
print("ID:", c["candidate_id"])
print("Name:", c["profile"]["anonymized_name"])
print("Title:", c["profile"]["current_title"])
print("Company:", c["profile"]["current_company"])
print("Industry:", c["profile"]["current_industry"])
print("Years of experience:", c["profile"]["years_of_experience"])
print("Location:", c["profile"]["location"], ",", c["profile"]["country"])

print("\n=== SUMMARY ===")
print(c["profile"]["summary"])

print("\n=== CAREER HISTORY ===")
for job in c["career_history"]:
    print(f"  - {job['title']} at {job['company']} ({job['duration_months']} months)")
    print(f"    {job['description'][:150]}...")

print("\n=== SKILLS ===")
for skill in c["skills"]:
    print(f"  - {skill['name']} | {skill['proficiency']} | {skill.get('duration_months', 0)} months")

print("\n=== BEHAVIORAL SIGNALS ===")
sig = c["redrob_signals"]
print("Open to work:", sig["open_to_work_flag"])
print("Last active:", sig["last_active_date"])
print("Recruiter response rate:", sig["recruiter_response_rate"])
print("Notice period (days):", sig["notice_period_days"])
print("Willing to relocate:", sig["willing_to_relocate"])
print("Github activity score:", sig["github_activity_score"])
print("Preferred work mode:", sig["preferred_work_mode"])