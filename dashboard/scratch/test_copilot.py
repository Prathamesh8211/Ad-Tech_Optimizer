import pandas as pd
import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from pages import _05_ai_copilot as ai_copilot
from llm import copilot

df = load_sample_data()
print("Loaded df shape:", df.shape)

# Simulate Agency view filtering (e.g. Device = Mobile)
df_filtered = df[df['ad_device'] == 'Mobile'].copy()
print("Filtered df shape (Mobile):", df_filtered.shape)

# Test extract_smart_context
q = "Identify the top performing campaigns with the highest ROAS."
ctx = copilot.extract_smart_context(q, df_filtered)
print("Context length:", len(ctx))

# Test render_simple_ai_response
ans = ai_copilot.render_simple_ai_response(q, df_filtered)
print("Answer sample:\n", ans[:200])
