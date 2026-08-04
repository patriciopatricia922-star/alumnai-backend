"""
debug_survey.py
---------------
Run directly in PyCharm terminal:
    python debug_survey.py

Shows exactly what train_model.py sees and whether each record
counts as "career aligned" using the correct two-condition logic.
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Service role key bypasses Row Level Security (RLS) so train_model.py
# can see ALL alumni rows, not just the ones the anon key is allowed to read.
# Never expose this key on the frontend -- backend/scripts only.
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)
if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
    print("  Using SERVICE ROLE key (bypasses RLS -- sees all rows)")
else:
    print("  WARNING: SUPABASE_SERVICE_ROLE_KEY not found in .env")
    print("  Falling back to ANON key -- RLS may hide rows!")
    print("  Add SUPABASE_SERVICE_ROLE_KEY to your .env file.")
    print()

supabase = create_client(os.getenv("SUPABASE_URL"), SUPABASE_KEY)

# Must be employed AND job related to degree to count as aligned
EMPLOYED_STATUSES = {
    "Regular / Permanent", "Contractual", "Probationary",
    "Part-time", "Self-employed", "Casual"
}
ALIGNED_JOB_VALUES = {
    "Yes", "yes", "YES",
    "Very much related", "Related", "Somewhat related",
    "1", "true", "True", "TRUE",
}

print("=" * 65)
print("  CAREER ALIGNMENT DIAGNOSTIC")
print("  Aligned = employed AND job related to degree")
print("=" * 65)

resp = supabase.table("survey_progress").select(
    "id, educational_background_data, employment_information_data"
).execute()

rows = resp.data or []
print(f"\nTotal rows in survey_progress: {len(rows)}\n")

usable = skipped = aligned_count = 0

for i, row in enumerate(rows, 1):
    edu = row.get("educational_background_data") or {}
    emp = row.get("employment_information_data") or {}

    degree          = edu.get("degree_program")
    status          = emp.get("employment_status")
    job_related_raw = emp.get("job_related_to_degree")

    print(f"--- Row {i}  (id: {str(row.get('id','N/A'))[:8]}...) ---")
    print(f"  degree_program:        {repr(degree)}")
    print(f"  employment_status:     {repr(status)}")
    print(f"  job_related_to_degree: {repr(job_related_raw)}")

    if not degree:
        print(f"  >> SKIPPED: degree_program is null/empty")
        skipped += 1
        print()
        continue

    is_employed = status in EMPLOYED_STATUSES if status else False
    is_related  = str(job_related_raw).strip() in ALIGNED_JOB_VALUES if job_related_raw else False
    aligned     = is_employed and is_related

    print(f"  is_employed:           {is_employed}")
    print(f"  is_job_related:        {is_related}")
    print(f"  >> CAREER ALIGNED:     {aligned}  ", end="")

    if aligned:
        print("(counts toward alignment %)")
        aligned_count += 1
    elif not is_employed:
        print("(unemployed -- does not count)")
    elif not is_related:
        print("(employed but job not related to degree -- does not count)")

    usable += 1
    print()

print("=" * 65)
print(f"  SUMMARY")
print(f"  Total rows:      {len(rows)}")
print(f"  Usable records:  {usable}")
print(f"  Skipped:         {skipped}  (missing degree_program)")
print(f"  Aligned:         {aligned_count}  (employed + job related to degree)")
if usable > 0:
    print(f"  Alignment rate:  {aligned_count/usable*100:.1f}%  across all programs")
print("=" * 65)
print()
print("NOTE: For a department card to appear, at least one alumni")
print("from a program in that department must have a survey row.")
print("If a department is missing, its alumni have not submitted")
print("the survey yet (no row exists in survey_progress).")