import pandas as pd
import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from pages import _05_ai_copilot as ai_copilot

df = load_sample_data()

q = "Identify the top performing Food, Gaming, Health, Travel campaigns with the highest ROAS."

resp = ai_copilot.render_simple_ai_response(q, df)
with open("scratch/crisp_out.txt", "w", encoding="utf-8") as f:
    f.write(resp)

print("SUCCESS: Crisp 3-block response written successfully.")
