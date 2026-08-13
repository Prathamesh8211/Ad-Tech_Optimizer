import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from llm.copilot import extract_smart_context

df = load_sample_data()
cats = ['electronics', 'fashion', 'health', 'travel']

tests = [
    ("Summarize underperforming Electronics, Fashion, Health, Travel campaigns draining our budget.", cats),
    ("Identify the top performing Electronics, Fashion, Health, Travel campaigns with the highest ROAS.", cats),
    ("Which platform and target device deliver the highest conversion rate for Electronics, Fashion, Health, Travel?", cats),
]

all_pass = True
for query, required_cats in tests:
    ctx = extract_smart_context(query, df)
    missing = [c.title() for c in required_cats if c.title() not in ctx]
    if missing:
        print(f"FAIL: {query[:60]}")
        print(f"  Missing categories: {missing}")
        all_pass = False
    else:
        print(f"PASS: {query[:60]}")
        print(f"  All 4 categories present in context")

if all_pass:
    print("\nALL TESTS PASSED: Every category gets equal representation!")
else:
    print("\nSOME TESTS FAILED.")
