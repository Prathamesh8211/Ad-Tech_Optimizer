import pandas as pd
import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from pages import _05_ai_copilot as ai_copilot
from llm import copilot

df = load_sample_data()

q = "Summarize underperforming Electronics, Fashion, Food, Gaming, Health campaigns draining our budget."

ctx = copilot.extract_smart_context(q, df)
with open("scratch/ctx_out.txt", "w", encoding="utf-8") as f:
    f.write(ctx)

resp = ai_copilot.render_simple_ai_response(q, df)
with open("scratch/multicat_out.txt", "w", encoding="utf-8") as f:
    f.write(resp)

print("SUCCESS: Context and Response written successfully.")
