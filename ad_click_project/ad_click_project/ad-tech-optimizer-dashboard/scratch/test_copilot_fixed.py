import pandas as pd
import sys
sys.path.append("c:\\Users\\nshra\\Downloads\\ad_click_project\\ad-tech-optimizer-dashboard")

from utils.data_loader import load_sample_data
from pages import _05_ai_copilot as ai_copilot
from llm import copilot

df = load_sample_data()

# Test Agency filtering
df_filtered = df[df['ad_device'] == 'Mobile'].copy()

# Test model detection
model = copilot.get_active_ollama_model()
print("Active detected Ollama model:", model)

# Test query
q = "What is the best device for Gaming?"
resp = ai_copilot.render_simple_ai_response(q, df_filtered)
with open("scratch/copilot_out.txt", "w", encoding="utf-8") as f:
    f.write(resp)

print("SUCCESS: Copilot response written to scratch/copilot_out.txt")
