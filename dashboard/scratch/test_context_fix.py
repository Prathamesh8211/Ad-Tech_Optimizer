import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from llm.copilot import extract_smart_context

df = load_sample_data()

tests = [
    ("Which platform and target device deliver the highest conversion rate for Electronics, Gaming, Health?",
     ["Electronics", "Gaming", "Health", "Performance by Device", "Performance by Platform", "$", "back per $1"]),
    ("Identify the top performing Food, Gaming campaigns with the highest ROAS.",
     ["Food", "Gaming", "Top Performing", "$", "back for every $1", "Spent $", "earned $"]),
    ("Summarize underperforming campaigns draining our budget.",
     ["Underperforming", "$", "per $1", "only earned"]),
]

all_passed = True
for query, must_contain in tests:
    ctx = extract_smart_context(query, df)
    failed = [m for m in must_contain if m not in ctx]
    if failed:
        print(f"FAIL for: {query[:60]}")
        print(f"  Missing: {failed}")
        print(f"  Context snippet:\n{ctx[:600]}\n")
        all_passed = False
    else:
        print(f"PASS: {query[:60]}")

# Also check that no raw ROAS x format sneaks into context (e.g. "2.55x")
import re
for query, _ in tests:
    ctx = extract_smart_context(query, df)
    roas_x_matches = re.findall(r'\d+\.\d+x', ctx)
    if roas_x_matches:
        print(f"  WARNING - raw ROAS 'Nx' format still in context for '{query[:40]}': {roas_x_matches[:5]}")
    else:
        print(f"  OK - No raw 'Nx' ROAS notation in context for '{query[:40]}'")

if all_passed:
    print("\nALL TESTS PASSED!")
